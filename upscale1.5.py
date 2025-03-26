import torch
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image, ImageEnhance
import threading
import time
import psutil
import gc
import pynvml
import os
import argparse
from huggingface_hub import login, snapshot_download

class MemoryMonitor:
    def __init__(self, num_gpus=2):
        self.keep_running = False
        self.thread = None
        self.num_gpus = num_gpus
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def get_memory_stats(self):
        # Get physical RAM stats
        ram = psutil.Process().memory_info()
        ram_total = psutil.virtual_memory().total / (1024 ** 3)
        ram_used = ram.rss / (1024 ** 3)

        # Get CPU usage
        cpu_percent = psutil.cpu_percent()

        stats = {
            'ram_used': ram_used,
            'ram_total': ram_total,
            'cpu_percent': cpu_percent,
            'gpu_stats': []
        }
        
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                # Get GPU stats
                gpu_used = torch.cuda.memory_allocated(i) / (1024 ** 3)
                gpu_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                
                # Get GPU utilization percentage
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handles[i]).gpu

                stats['gpu_stats'].append({
                    'device': f'cuda:{i}',
                    'gpu_used': gpu_used,
                    'gpu_total': gpu_total,
                    'gpu_util': gpu_util
                })
        
        return stats

    def monitor_memory(self):
        while self.keep_running:
            stats = self.get_memory_stats()
            
            gpu_info_list = []
            for gpu_stat in stats['gpu_stats']:
                vram_info = f"VRAM {gpu_stat['device']}: {gpu_stat['gpu_used']:.1f}GB/{gpu_stat['gpu_total']:.1f}GB"
                gpu_info = f"Usage: {gpu_stat['gpu_util']}%"
                gpu_info_list.append(f"{vram_info}, {gpu_info}")
                
            ram_info = f"RAM: {stats['ram_used']:.1f}GB/{stats['ram_total']:.1f}GB"
            cpu_info = f"CPU: {stats['cpu_percent']}%"
            
            print(f"{' | '.join(gpu_info_list)} | {ram_info} | {cpu_info}")
            
            time.sleep(3)

    def start_monitoring(self):
        self.keep_running = True
        self.thread = threading.Thread(target=self.monitor_memory)
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        self.keep_running = False
        if self.thread:
            self.thread.join()

class FluxUpscaler:
    def __init__(self, 
                 main_gpu="cuda:0", 
                 controlnet_gpu="cuda:1", 
                 model_id="black-forest-labs/FLUX.1-dev", 
                 controlnet_id="jasperai/Flux.1-dev-Controlnet-Upscaler", 
                 hf_token=None):
        
        self.memory_monitor = MemoryMonitor(num_gpus=torch.cuda.device_count())
        self.hf_token = hf_token
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.main_gpu = main_gpu
        self.controlnet_gpu = controlnet_gpu
            
        print(f"Using main GPU: {self.main_gpu}, ControlNet GPU: {self.controlnet_gpu}")
        self.setup()

    def apply_cc_effects(self, img: Image.Image) -> Image.Image:
        print("Applying CC effects")
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Adjust exposure
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.60)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)
        
        # Adjust vibrance/saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.06)
        
        return img

    def print_system_info(self):
        print("\nSystem Information:")
        # Print CPU information
        try:
            import platform
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            print(f"CPU: {cpu_name}")
                            break
            elif platform.system() == "Windows":
                import subprocess
                cpu_info = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split('\n')[1]
                print(f"CPU: {cpu_info}")
        except:
            print("CPU: Information not available")

        # Print GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"GPU {i}: {gpu_name} ({gpu_total_mem:.1f}GB)")
        else:
            print("GPU: Not available")

    def setup(self) -> None:
        print("\nLoading the model into memory with split GPU configuration...")
        
        # Set PyTorch memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Clear CUDA cache before loading models
        torch.cuda.empty_cache()
        
        # Authenticate with Hugging Face
        if self.hf_token:
            login(token=self.hf_token)
            print("HF token authenticated")
        else:
            print("Warning: No HF token provided, might have limited access to models")

        # Download model
        model_path = snapshot_download(
            repo_id=self.model_id,
            repo_type="model",
            ignore_patterns=["*.md", "*.gitattributes"],
            local_dir="FLUX.1-dev",
            token=self.hf_token,
        )
        
        # CRITICAL CHANGE: Load the controlnet first
        print(f"Loading ControlNet on {self.controlnet_gpu}")
        torch.cuda.set_device(int(self.controlnet_gpu.split(':')[1]))
        
        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.bfloat16
        ).to(self.controlnet_gpu)
        
        # Then load the main pipeline with the controlnet reference
        print(f"Loading main model components on {self.main_gpu}")
        torch.cuda.set_device(int(self.main_gpu.split(':')[1]))
        
        # We'll first create a dummy placeholder controlnet on the main device
        # This is just for initialization - we'll replace it with proper references later
        placeholder_controlnet = None
        
        # Now load the main pipeline, providing all required components
        self.pipe = FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,  # Pass the already loaded controlnet
            torch_dtype=torch.bfloat16
        )
        
        # Move all other components to main GPU
        for name, component in self.pipe.components.items():
            if name == "transformer" and hasattr(component, "to"):
                print(f"Moving {name} to {self.controlnet_gpu}")
                self.pipe.components[name] = component.to(self.controlnet_gpu)
            elif name != "controlnet" and hasattr(component, "to"):
                print(f"Moving {name} to {self.main_gpu}")
                self.pipe.components[name] = component.to(self.main_gpu)
        
        # Keep explicit reference to controlnet
        self.controlnet = controlnet
        
        # Enable memory efficient attention if available
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            print("Enabling xformers memory efficient attention")
            self.pipe.enable_xformers_memory_efficient_attention()
        
        # Add handler for cross-device operations
        self._add_controlnet_device_handler()
        
        print("Model successfully loaded across GPUs")

    def _add_controlnet_device_handler(self):
        """Add a handler to manage cross-device tensor operations for ControlNet"""
        
        # Store the original forward method
        original_forward = self.controlnet.forward
        main_gpu = self.main_gpu
        
        # Define a new forward method that handles device transitions
        def patched_forward(self, *args, **kwargs):
            # Get input tensor that may need device transition
            sample = kwargs.get('sample', None)
            if sample is None and len(args) > 0:
                sample = args[0]
            
            # Move the sample tensor to controlnet's device if needed
            device_moved = False
            if sample is not None and hasattr(sample, 'device') and sample.device != self.device:
                print(f"ControlNet: Moving input tensor from {sample.device} to {self.device}")
                sample_device = sample.device  # Remember original device
                sample = sample.to(self.device)
                device_moved = True
                
                # Update the args or kwargs depending on where sample was
                if len(args) > 0:
                    args = (sample,) + args[1:]
                else:
                    kwargs['sample'] = sample
            
            # Call the original forward pass
            output = original_forward(self, *args, **kwargs)
            
            # Move output back to original device if we moved the input
            if device_moved:
                if isinstance(output, tuple):
                    # Move each tensor in the tuple
                    output = tuple(o.to(main_gpu) if torch.is_tensor(o) else o for o in output)
                elif torch.is_tensor(output):
                    # Move single tensor output
                    output = output.to(main_gpu)
            
            return output
        
        # Apply our patched method
        self.controlnet.forward = patched_forward.__get__(self.controlnet, type(self.controlnet))
        
        print(f"Added cross-device handler for ControlNet ({self.controlnet_gpu} <-> {self.main_gpu})")

    def process_input(self, input_image: Image.Image) -> Image.Image:
        print("Processing input image dimensions")
        # Get original dimensions
        w, h = input_image.size
        aspect_ratio = h / w
        
        # Handle width conditions
        if w > 576:
            # Downscale if width is too large
            w = 576
            h = int(w * aspect_ratio)
            input_image = input_image.resize((w, h), Image.Resampling.LANCZOS)
        elif w < 240:
            # Upscale if width is too small
            w = 480
            h = int(w * aspect_ratio)
            input_image = input_image.resize((w, h), Image.Resampling.LANCZOS)
            
        # Ensure dimensions are multiple of 8 for model compatibility
        w = w - w % 8
        h = h - h % 8
        return input_image.resize((w, h), Image.Resampling.LANCZOS)

    def upscale(
        self,
        input_image_path,
        output_path="output.jpg",
        prompt="",
        guidance_scale=5.0,
        apply_cc_preset=False,
        num_inference_steps=28,
        upscale_factor=4,
        controlnet_conditioning_scale=0.6,
        seed=None,
    ):
        """Run a single upscaling process"""
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        self.print_system_info()
        print(f"Seed: {seed}")
        
        input_image = Image.open(input_image_path).convert("RGB")
        # Store original dimensions
        original_size = input_image.size
        
        input_image = self.process_input(input_image)
        print(f"Processed input dimensions: {input_image.size}")

        # Prepare control image
        w, h = input_image.size
        upscaled_w, upscaled_h = w * upscale_factor, h * upscale_factor
        print(f"Target upscaled dimensions: {upscaled_w}x{upscaled_h}")
        control_image = input_image.resize((upscaled_w, upscaled_h), Image.Resampling.LANCZOS)

        # Create generator on main device
        generator = torch.Generator(device=self.main_gpu).manual_seed(seed)

        print("Upscaling started, monitoring memory usage...")
        self.memory_monitor.start_monitoring()
        try:
            # Clear CUDA cache before running inference
            torch.cuda.empty_cache()
            
            # Run the pipeline
            output = self.pipe(
                prompt=prompt,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=upscaled_h,
                width=upscaled_w,
                generator=generator,
            )
            
            output_image = output.images[0]
            
        finally:
            print("Stopping memory monitoring...")
            self.memory_monitor.stop_monitoring()
            gc.collect()
            torch.cuda.empty_cache()

        # Apply CC effects to upscaled image if enabled
        if apply_cc_preset:
            output_image = self.apply_cc_effects(output_image)

        # Save output image
        print(f"Saving output image to {output_path}")
        output_image.save(output_path)
        print("Process complete!")
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FLUX Upscaler with Multi-GPU Support')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', default='0.jpg', help='Output image path')
    parser.add_argument('--prompt', default='', help='Text prompt to guide upscaling')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--cc', action='store_true', help='Apply color correction')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor (1-4)')
    parser.add_argument('--ccs', type=float, default=0.6, help='ControlNet conditioning scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--main_gpu', default="cuda:0", help='Device for main model')
    parser.add_argument('--controlnet_gpu', default="cuda:1", help='Device for controlnet')
    parser.add_argument('--token', default="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", help='HuggingFace token')
    
    args = parser.parse_args()
    
    upscaler = FluxUpscaler(
        main_gpu=args.main_gpu, 
        controlnet_gpu=args.controlnet_gpu, 
        hf_token=args.token
    )
    
    upscaler.upscale(
        input_image_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        apply_cc_preset=args.cc,
        num_inference_steps=args.steps,
        upscale_factor=args.upscale,
        controlnet_conditioning_scale=args.ccs,
        seed=args.seed,
    )