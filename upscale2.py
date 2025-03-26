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
                hf_token=None,
                offload_vae=True,           # Offload VAE to CPU when not in use
                use_float16=True,           # Force float16 precision
                use_sequential_cpu_offload=False,  # For extreme memory constraints
                use_attention_slicing=True,  # Slice attention computation
                use_model_sharding=True     # Whether to use model sharding for transformer
                ):
        
        self.memory_monitor = MemoryMonitor(num_gpus=torch.cuda.device_count())
        self.hf_token = hf_token
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.main_gpu = main_gpu
        self.controlnet_gpu = controlnet_gpu
        self.offload_vae = offload_vae
        self.use_float16 = use_float16
        self.use_sequential_cpu_offload = use_sequential_cpu_offload
        self.use_attention_slicing = use_attention_slicing
        self.use_model_sharding = use_model_sharding
        
        # Use more memory-efficient precision
        self.torch_dtype = torch.float16 if use_float16 else torch.bfloat16
        
        print(f"Using main GPU: {self.main_gpu}, ControlNet GPU: {self.controlnet_gpu}")
        print(f"Using torch dtype: {self.torch_dtype}")
        
        # Set PyTorch memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
            
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

    def setup(self):
        print("\nLoading the model into memory with transformer-split GPU configuration...")
        
        # Clear CUDA cache before loading models
        torch.cuda.empty_cache()
        gc.collect()
        
        # Authenticate with Hugging Face if token provided
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
        
        # Load ControlNet first on second GPU since it's smaller
        print(f"Loading ControlNet on {self.controlnet_gpu}")
        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.torch_dtype,
        ).to(self.controlnet_gpu)
        
        # Calculate available memory on both GPUs to inform device map
        torch.cuda.empty_cache()
        gpu0_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        gpu1_free = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_allocated(1)
        
        print(f"Free memory - GPU 0: {gpu0_free / 1e9:.2f}GB, GPU 1: {gpu1_free / 1e9:.2f}GB")

        # Use model sharding if requested
        if self.use_model_sharding:
            try:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                print("Using model sharding for transformer...")
                
                # First load the pipeline without the transformer
                device_map = {
                    "text_encoder": self.main_gpu,
                    "text_encoder_2": self.main_gpu,
                    "tokenizer": "cpu",
                    "tokenizer_2": "cpu",
                    "scheduler": "cpu",
                    "feature_extractor": "cpu",
                    "controlnet": self.controlnet_gpu,
                    "vae": self.main_gpu if not self.offload_vae else "cpu",
                    # Transformer will be loaded separately
                    "transformer": None
                }
                
                # Load pipeline without transformer
                self.pipe = FluxControlNetPipeline.from_pretrained(
                    model_path,
                    controlnet=controlnet,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                )
                
                # Now load transformer with sharding
                with init_empty_weights():
                    # Load config only
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(os.path.join(model_path, "transformer"))
                    # Create empty model
                    from transformers import AutoModelForCausalLM
                    transformer_empty = AutoModelForCausalLM.from_config(config)
                
                # Set up device map - includes both GPUs and CPU
                max_memory = {
                    0: "24GB",  # First GPU
                    1: "24GB",  # Second GPU (give it more since controlnet is smaller)
                    "cpu": "48GB"  # Allow CPU offloading
                }
                
                # Load checkpoint with weights distributed according to available memory
                self.pipe.transformer = load_checkpoint_and_dispatch(
                    transformer_empty, 
                    os.path.join(model_path, "transformer"), 
                    device_map="auto",
                    max_memory=max_memory,
                    offload_folder="offload",
                    offload_state_dict=True
                )
            except ImportError:
                print("Error: accelerate library not available for model sharding. Falling back to manual device mapping.")
                self.use_model_sharding = False
        
        if not self.use_model_sharding:
            # Define specific layers for transformer to place on each GPU
            # This is critical since the whole transformer won't fit on one GPU
            transformer_device_map = {
                # Split transformer model across GPUs based on layers
                # First half on GPU 0
                "text_projection": self.main_gpu,
                "text_projection_2": self.main_gpu,
                "positional_embedding": self.main_gpu, 
                "final_ln": self.main_gpu,
                "transformer.resblocks.0": self.main_gpu,
                "transformer.resblocks.1": self.main_gpu,
                "transformer.resblocks.2": self.main_gpu,
                "transformer.resblocks.3": self.main_gpu,
                "transformer.resblocks.4": self.main_gpu,
                "transformer.resblocks.5": self.main_gpu,
                # Second half on GPU 1
                "transformer.resblocks.6": self.controlnet_gpu,
                "transformer.resblocks.7": self.controlnet_gpu,
                "transformer.resblocks.8": self.controlnet_gpu,
                "transformer.resblocks.9": self.controlnet_gpu,
                "transformer.resblocks.10": self.controlnet_gpu,
                "transformer.resblocks.11": self.controlnet_gpu,
            }
            
            # Create device map for the whole pipeline
            device_map = {
                "text_encoder": self.main_gpu,
                "text_encoder_2": self.main_gpu,
                "tokenizer": "cpu",
                "tokenizer_2": "cpu",
                "scheduler": "cpu",
                "feature_extractor": "cpu",
                "controlnet": self.controlnet_gpu,
                "vae": self.main_gpu if not self.offload_vae else "cpu",
            }
            
            # Add transformer device mapping
            for key, value in transformer_device_map.items():
                device_map[f"transformer.{key}"] = value
            
            # Load the full pipeline with our device mapping
            print("Loading pipeline with custom transformer device mapping...")
            self.pipe = FluxControlNetPipeline.from_pretrained(
                model_path,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                variant="fp16" if self.use_float16 else None,
            )

        # Enable memory efficient attention mechanisms
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            print("Enabling xformers memory efficient attention")
            self.pipe.enable_xformers_memory_efficient_attention()
        elif hasattr(self.pipe, 'enable_attention_slicing') and self.use_attention_slicing:
            print("Enabling attention slicing")
            self.pipe.enable_attention_slicing(slice_size="auto")
        
        # Enable flash attention if available
        if hasattr(self.pipe, "enable_flash_attention"):
            print("Enabling flash attention")
            self.pipe.enable_flash_attention()

        # Enable sequential CPU offload if requested
        if self.use_sequential_cpu_offload:
            from accelerate import cpu_offload_with_hook
            print("Enabling sequential CPU offloading")
            for name, module in self.pipe.components.items():
                if name not in ["scheduler", "safety_checker"]:
                    cpu_offload_with_hook(module, execution_device=self.main_gpu if name != "controlnet" else self.controlnet_gpu)

        # Keep controlnet reference
        self.controlnet = self.pipe.controlnet
        
        # Add handler for cross-device operations
        self._add_cross_device_handlers()
        
        print("Model successfully loaded with transformer split across GPUs")

    def _add_cross_device_handlers(self):
        """Add handlers to manage tensors moving between devices for split models"""
        print("Setting up cross-device handlers for split model components...")
        
        # Patch controlnet forward method
        if self.controlnet is not None:
            original_controlnet_forward = self.controlnet.forward
            
            def patched_controlnet_forward(self, *args, **kwargs):
                # Handle input tensors to ensure they're on the right device
                if 'sample' in kwargs and torch.is_tensor(kwargs['sample']):
                    kwargs['sample'] = kwargs['sample'].to(self.device)
                elif args and torch.is_tensor(args[0]):
                    args = list(args)
                    args[0] = args[0].to(self.device)
                    args = tuple(args)
                
                # Call original forward method
                output = original_controlnet_forward(self, *args, **kwargs)
                
                # Move output back to main device if needed
                if isinstance(output, tuple):
                    output = tuple(
                        tensor.to("cuda:0") if torch.is_tensor(tensor) and hasattr(tensor, 'device') else tensor
                        for tensor in output
                    )
                elif torch.is_tensor(output) and hasattr(output, 'device'):
                    output = output.to("cuda:0")
                
                return output
            
            # Apply the patched method
            self.controlnet.forward = patched_controlnet_forward.__get__(
                self.controlnet, type(self.controlnet)
            )
        
        # Handle transformer attention blocks if needed
        if not self.use_model_sharding and hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
            # This targets specific modules that might need cross-device handling
            module_patterns = ["attn", "ln_", "resblocks"]
            
            for name, module in self.pipe.transformer.named_modules():
                # Only patch modules that match our patterns and have a forward method
                if any(pattern in name.lower() for pattern in module_patterns) and hasattr(module, "forward"):
                    original_forward = module.forward
                    
                    def make_patched_forward(orig_forward, module_device):
                        def patched_forward(self, *args, **kwargs):
                            # Move key inputs to this device
                            moved_tensors = {}
                            for k, v in kwargs.items():
                                if torch.is_tensor(v) and hasattr(v, 'device') and v.device.type == 'cuda' and v.device != module_device:
                                    moved_tensors[k] = v.to(module_device)
                                    kwargs[k] = moved_tensors[k]
                            
                            # Handle positional args
                            new_args = []
                            for arg in args:
                                if torch.is_tensor(arg) and hasattr(arg, 'device') and arg.device.type == 'cuda' and arg.device != module_device:
                                    new_args.append(arg.to(module_device))
                                else:
                                    new_args.append(arg)
                            
                            # Call original method
                            output = orig_forward(self, *new_args, **kwargs)
                            
                            # We generally don't move outputs back to avoid overhead
                            # Let the next module handle it if needed
                            return output
                        
                        return patched_forward
                    
                    # Get the device for this module
                    try:
                        module_device = next(module.parameters()).device
                        module.forward = make_patched_forward(original_forward, module_device).__get__(
                            module, type(module)
                        )
                    except (StopIteration, RuntimeError):
                        # Skip modules with no parameters
                        pass
        
        print("Cross-device handlers configured for split model")

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
        """Run a single upscaling process with aggressive memory management"""
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        self.print_system_info()
        print(f"Seed: {seed}")
        
        # Process input image
        input_image = Image.open(input_image_path).convert("RGB")
        original_size = input_image.size
        input_image = self.process_input(input_image)
        print(f"Processed input dimensions: {input_image.size}")
        
        # Prepare control image with smaller intermediate size if needed
        w, h = input_image.size
        upscaled_w, upscaled_h = w * upscale_factor, h * upscale_factor
        print(f"Target upscaled dimensions: {upscaled_w}x{upscaled_h}")
        
        # For very large upscales, consider using a two-stage approach
        large_upscale = (upscaled_w * upscaled_h) > (2048 * 2048)
        if large_upscale:
            print("Large output detected, using two-stage upscaling...")
            intermediate_factor = 2
            # First upscale to intermediate size
            intermediate_w, intermediate_h = w * intermediate_factor, h * intermediate_factor
            control_image = input_image.resize((intermediate_w, intermediate_h), Image.Resampling.LANCZOS)
            print(f"Intermediate dimensions: {intermediate_w}x{intermediate_h}")
        else:
            control_image = input_image.resize((upscaled_w, upscaled_h), Image.Resampling.LANCZOS)
        
        # Create generator
        generator = torch.Generator(device=self.main_gpu).manual_seed(seed)
        
        print("Upscaling started, monitoring memory usage...")
        self.memory_monitor.start_monitoring()
        
        try:
            # Clear cache before inference
            torch.cuda.empty_cache()
            gc.collect()
            
            # Move VAE to CPU temporarily to free up memory if configured
            if self.offload_vae and hasattr(self.pipe, 'vae'):
                vae_device = self.pipe.vae.device
                self.pipe.vae = self.pipe.vae.to("cpu")
            
            # Setup inference parameters with memory optimizations
            inference_kwargs = {
                "prompt": prompt,
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": intermediate_h if large_upscale else upscaled_h,
                "width": intermediate_w if large_upscale else upscaled_w,
                "generator": generator,
            }
            
            # Enable chunked processing for large images
            if hasattr(self.pipe, "enable_chunked_generation"):
                print("Enabling chunked generation for memory efficiency")
                self.pipe.enable_chunked_generation()
            
            # Aggressive memory optimization
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.pipe(**inference_kwargs)
            
            # Move VAE back for decoding if it was offloaded
            if self.offload_vae and hasattr(self.pipe, 'vae'):
                self.pipe.vae = self.pipe.vae.to(vae_device)
                
            output_image = output.images[0]
            
            # For large upscales, do second stage
            if large_upscale:
                print("Stage 1 complete. Running final upscale stage...")
                # Use a traditional upscaler for final stage
                output_image = output_image.resize(
                    (upscaled_w, upscaled_h), 
                    Image.Resampling.LANCZOS
                )
        
        finally:
            self.memory_monitor.stop_monitoring()
            # Aggressively clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        # Apply CC effects if enabled
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
    parser.add_argument('--offload_vae', action='store_true', help='Offload VAE to CPU when not in use')
    parser.add_argument('--float16', action='store_true', help='Use float16 precision')
    parser.add_argument('--cpu_offload', action='store_true', help='Use sequential CPU offloading')
    parser.add_argument('--attention_slicing', action='store_true', help='Use attention slicing')
    parser.add_argument('--model_sharding', action='store_true', help='Use model sharding for transformer')
    
    args = parser.parse_args()
    
    upscaler = FluxUpscaler(
        main_gpu=args.main_gpu, 
        controlnet_gpu=args.controlnet_gpu, 
        hf_token=args.token,
        offload_vae=args.offload_vae,
        use_float16=args.float16,
        use_sequential_cpu_offload=args.cpu_offload,
        use_attention_slicing=args.attention_slicing,
        use_model_sharding=args.model_sharding
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