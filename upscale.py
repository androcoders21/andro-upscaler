import torch
import numpy as np
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image, ImageEnhance
import threading
import time
import psutil
import gc
import pynvml
import argparse
from pathlib import Path

class MemoryMonitor:
    def __init__(self):
        self.keep_running = False
        self.thread = None
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_memory_stats(self):
        ram = psutil.Process().memory_info()
        ram_total = psutil.virtual_memory().total / (1024 ** 3)
        ram_used = ram.rss / (1024 ** 3)
        cpu_percent = psutil.cpu_percent()

        stats = {
            'ram_used': ram_used,
            'ram_total': ram_total,
            'cpu_percent': cpu_percent
        }
        
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            stats.update({
                'gpu_used': gpu_used,
                'gpu_total': gpu_total,
                'gpu_util': gpu_util
            })
        
        return stats

    def monitor_memory(self):
        while self.keep_running:
            stats = self.get_memory_stats()
            if torch.cuda.is_available():
                vram_info = f"VRAM: {stats['gpu_used']:.1f}GB/{stats['gpu_total']:.1f}GB"
                gpu_info = f"GPU Usage: {stats['gpu_util']}%"
            else:
                vram_info = "VRAM: N/A"
                gpu_info = "GPU: N/A"
                
            ram_info = f"Physical RAM: {stats['ram_used']:.1f}GB/{stats['ram_total']:.1f}GB"
            cpu_info = f"CPU: {stats['cpu_percent']}%"
            
            print(f"{vram_info}, {gpu_info}, {ram_info}, {cpu_info}")
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

class ImageUpscaler:
    def print_gpu_memory(self):
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            used = torch.cuda.memory_allocated(i) / 1e9
            print(f"GPU {i}: Using {used:.1f}GB / {total:.1f}GB")

    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        # Clear any existing cache
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            print(f"\nFound {torch.cuda.device_count()} GPUs!")
            self.print_gpu_memory()
        self.setup()

    def apply_cc_effects(self, img: Image.Image) -> Image.Image:
        print("Applying CC effects")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.60)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.06)
        
        return img

    def print_system_info(self):
        print("\nSystem Information:")
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
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                print(f"CPU: {cpu_name}")
                winreg.CloseKey(key)
        except:
            print("CPU: Information not available")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU: {gpu_name} ({gpu_total_mem:.1f}GB)")
        else:
            print("GPU: Not available")

    def setup(self):
        print("\nLoading the model into memory")
        if torch.cuda.device_count() > 1:
            print(f"Enabling multi-GPU support with {torch.cuda.device_count()} GPUs")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Authenticating with HuggingFace")
        from huggingface_hub import login, snapshot_download
        
        hf_token = "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"
        if hf_token:
            login(token=hf_token)

        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            repo_type="model",
            ignore_patterns=["*.md", "*.gitattributes"],
            local_dir="FLUX.1-dev",
            token=hf_token,
        )
        
        print("Loading pipeline components")
        # Load controlnet
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Enable multi-GPU for controlnet if available
        if torch.cuda.device_count() > 1:
            self.controlnet = torch.nn.DataParallel(self.controlnet)
        
        # Load pipeline
        self.pipe = FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16 
        ).to(self.device)
        
        # Enable multi-GPU for pipeline components if available
        if torch.cuda.device_count() > 1:
            self.pipe.unet = torch.nn.DataParallel(self.pipe.unet)
            
        # Print GPU memory usage after loading
        if torch.cuda.is_available():
            print("\nGPU memory usage after model loading:")
            self.print_gpu_memory()

    def process_input(self, input_image: Image.Image) -> Image.Image:
        print("Processing input image dimensions")
        w, h = input_image.size
        aspect_ratio = h / w
        
        if w > 576:
            w = 576
            h = int(w * aspect_ratio)
            input_image = input_image.resize((w, h), Image.LANCZOS)
        elif w < 240:
            w = 480
            h = int(w * aspect_ratio)
            input_image = input_image.resize((w, h), Image.LANCZOS)
            
        w = w - w % 8
        h = h - h % 8
        return input_image.resize((w, h), Image.LANCZOS)

    def upscale(self, input_path: str, output_path: str = "output.jpg", prompt: str = "",
                guidance_scale: float = 5.0, apply_cc_preset: bool = False,
                num_inference_steps: int = 28, upscale_factor: int = 4,
                controlnet_conditioning_scale: float = 0.6, seed: int = None) -> str:
        
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        self.print_system_info()
        print(f"Seed: {seed}")
        
        input_image = Image.open(input_path).convert("RGB")
        original_size = input_image.size
        input_image = self.process_input(input_image)

        w, h = input_image.size
        control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Upscaling Started, Starting memory monitoring...")
        self.memory_monitor.start_monitoring()
        try:
            output_image = self.pipe(
                prompt=prompt,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=control_image.size[1],
                width=control_image.size[0],
                generator=generator,
            ).images[0]
        finally:
            print("Stopping memory monitoring...")
            self.memory_monitor.stop_monitoring()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if apply_cc_preset:
            output_image = self.apply_cc_effects(output_image)

        print("Saving output image")
        output_path = Path(output_path)
        output_image = output_image.resize(original_size, Image.LANCZOS)
        output_image.save(output_path)
        
        # Print final GPU memory usage
        if torch.cuda.is_available():
            print("\nFinal GPU memory usage:")
            self.print_gpu_memory()
            
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="Image Upscaler using FLUX.1")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--output", default="output.jpg", help="Path for the output image")
    parser.add_argument("--prompt", default="", help="Text prompt to guide the upscaling")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Guidance scale (1.0-20.0)")
    parser.add_argument("--apply-cc", action="store_true", help="Apply color correction")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (8-50)")
    parser.add_argument("--upscale-factor", type=int, default=4, help="Upscale factor (1-4)")
    parser.add_argument("--controlnet-scale", type=float, default=0.6, help="ControlNet conditioning scale (0.1-1.5)")
    parser.add_argument("--seed", type=int, help="Random seed (optional)")

    args = parser.parse_args()

    upscaler = ImageUpscaler()
    output_path = upscaler.upscale(
        args.input_image,
        args.output,
        args.prompt,
        args.guidance_scale,
        args.apply_cc,
        args.steps,
        args.upscale_factor,
        args.controlnet_scale,
        args.seed
    )
    print(f"Upscaled image saved to: {output_path}")

if __name__ == "__main__":
    main()
