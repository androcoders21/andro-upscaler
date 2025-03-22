from typing import List
import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image, ImageEnhance, ImageFilter
import warnings
import threading
import time
import psutil
import gc
import pynvml

class MemoryMonitor:
    def __init__(self):
        self.keep_running = False
        self.thread = None
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_memory_stats(self):
        # Get physical RAM stats (RSS - Resident Set Size)
        ram = psutil.Process().memory_info()
        ram_total = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
        ram_used = ram.rss / (1024 ** 3)  # Physical memory used

        # Get CPU usage
        cpu_percent = psutil.cpu_percent()

        stats = {
            'ram_used': ram_used,
            'ram_total': ram_total,
            'cpu_percent': cpu_percent
        }
        
        if torch.cuda.is_available():
            # Get GPU stats
            gpu_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            
            # Get GPU utilization percentage
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

class Predictor(BasePredictor):
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
    def apply_cc_effects(self, img: Image.Image) -> Image.Image:
        print("Applying CC effects")
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Adjust exposure (-1 in Lightroom scale ≈ -15% brightness)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.60)
        
        # Adjust contrast (+25 in Lightroom scale ≈ 25% increase)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)
        
        # Adjust vibrance (+16 in Lightroom scale ≈ subtle saturation boost)
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
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                print(f"CPU: {cpu_name}")
                winreg.CloseKey(key)
        except:
            print("CPU: Information not available")

        # Print GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU: {gpu_name} ({gpu_total_mem:.1f}GB)")
        else:
            print("GPU: Not available")

    def setup(self) -> None:
        print("\nLoading the model into memory to make running multiple predictions efficient")
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print("HF token authenticated")
        from huggingface_hub import login, snapshot_download
        import os
        
        # Set token directly here
        hf_token = "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"
        if hf_token:
            login(token =hf_token)

        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            repo_type="model",
            ignore_patterns=["*.md", "*.gitattributes"],
            local_dir="FLUX.1-dev",
            token=hf_token,
        )

        
        print("Loading pipeline components")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.pipe = FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16 
        ).to(self.device)

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
            input_image = input_image.resize((w, h), Image.LANCZOS)
        elif w < 240:
            # Upscale if width is too small
            w = 480
            h = int(w * aspect_ratio)
            input_image = input_image.resize((w, h), Image.LANCZOS)
            
        # Ensure dimensions are multiple of 8 for model compatibility
        w = w - w % 8
        h = h - h % 8
        return input_image.resize((w, h), Image.LANCZOS)

    def predict(
        self,
        input_image: Path = Input(description="Input image to upscale"),
        prompt: str = Input(
            description="Text prompt to guide the upscaling. Leave empty for default behavior.",
            default="",
        ),
         guidance_scale: float = Input(
            description="Guidance scale for Image adherence",
            default=5.0,
            ge=1.0,
            le=20.0,
        ),
         apply_cc_preset: bool = Input(
            description="Applying Color Correction ",
            default=False,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=28,
            ge=8,
            le=50,
        ),
        upscale_factor: int = Input(
            description="Factor by which to upscale the image",
            default=4,
            ge=1,
            le=4,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Controlnet conditioning scale",
            default=0.6,
            ge=0.1,
            le=1.5,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        self.print_system_info()
        print(f"Seed: {seed}")
        
        input_image = Image.open(input_image).convert("RGB")
        # Store original dimensions
        original_size = input_image.size
        
        input_image = self.process_input(input_image)

        # Prepare control image
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

        # Apply CC effects to upscaled image if enabled
        if apply_cc_preset:
            output_image = self.apply_cc_effects(output_image)

        # Resize back to original dimensions and save
        print("Saving output image")
        output_path = Path("output.jpg")
        output_image = output_image.resize(original_size, Image.LANCZOS)
        output_image.save(output_path)
        return output_path
