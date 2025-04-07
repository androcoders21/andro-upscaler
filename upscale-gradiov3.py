from typing import Optional, Tuple
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
import platform
import gradio as gr
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self):
        self.keep_running = False
        self.thread = None
        self.memory_stats = {}  # For storing the latest stats
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
        
        self.memory_stats = stats  # Store the latest stats
        return stats

    def monitor_memory(self, status_callback=None):
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
            
            status_msg = f"{vram_info}, {gpu_info}, {ram_info}, {cpu_info}"
            logger.info(status_msg)
            
            # Update UI if callback provided
            if status_callback:
                status_callback(status_msg)
            
            time.sleep(3)

    def start_monitoring(self, status_callback=None):
        self.keep_running = True
        self.thread = threading.Thread(target=self.monitor_memory, args=(status_callback,))
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        self.keep_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

class ImageUpscaler:
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.controlnet = None
        self.pipe = None
        self.model_loaded = False
    
    def apply_cc_effects(self, img: Image.Image) -> Image.Image:
        logger.info("Applying CC effects")
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

    def get_system_info(self) -> str:
        info = ["System Information:"]
        
        # CPU information
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            info.append(f"CPU: {cpu_name}")
                            break
            elif platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                info.append(f"CPU: {cpu_name}")
                winreg.CloseKey(key)
        except Exception as e:
            info.append(f"CPU: Information not available ({str(e)})")

        # GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info.append(f"GPU: {gpu_name} ({gpu_total_mem:.1f}GB)")
        else:
            info.append("GPU: Not available")
            
        return "\n".join(info)

    def load_model(self, status_callback=None) -> Tuple[bool, str]:
        if status_callback:
            status_callback("Loading model into memory...")
        
        try:
            from huggingface_hub import login, snapshot_download
            
            # Set token directly here (consider using environment variable in production)
            hf_token = "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"
            if hf_token:
                login(token=hf_token)
                if status_callback:
                    status_callback("HF token authenticated")
            else:
                if status_callback:
                    status_callback("No HF token provided, using public access")

            os.makedirs("FLUX.1-dev", exist_ok=True)
            model_path = snapshot_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                repo_type="model",
                ignore_patterns=["*.md", "*.gitattributes"],
                local_dir="FLUX.1-dev",
                token=hf_token,
            )
            
            if status_callback:
                status_callback("Loading pipeline components...")
            
            self.controlnet = FluxControlNetModel.from_pretrained(
                "jasperai/Flux.1-dev-Controlnet-Upscaler",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            self.pipe = FluxControlNetPipeline.from_pretrained(
                model_path,
                controlnet=self.controlnet,
                torch_dtype=torch.bfloat16 
            ).to(self.device)
            
            self.model_loaded = True
            return True, "Model loaded successfully!"
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def upscale_image(
        self,
        input_image,
        prompt="",
        guidance_scale=5.0,
        apply_cc_preset=False,
        num_inference_steps=28,
        upscale_factor=4,
        controlnet_conditioning_scale=0.6,
        seed=None,
        status_callback=None,
    ) -> Tuple[Optional[Image.Image], str]:
        if not self.model_loaded:
            return None, "Model not loaded. Please load the model first."
        
        if input_image is None:
            return None, "No input image provided."
        
        try:
            # Generate seed if not provided
            if seed is None or seed == 0:
                seed = int(torch.randint(0, 1000000, (1,))[0].item())
            
            system_info = self.get_system_info()
            if status_callback:
                status_callback(f"{system_info}\nSeed: {seed}")
            
            # Ensure input is PIL Image
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            
            input_image = input_image.convert("RGB")
            
            # Prepare control image
            w, h = input_image.size
            
            # Calculate scaling factor to fit within 1280px bounds
            scale = min(1280 / w, 1280 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Round to nearest multiple of 16
            new_w = round(new_w / 16) * 16
            new_h = round(new_h / 16) * 16
            
            if status_callback:
                status_callback(f"Resizing image to {new_w}x{new_h} (multiples of 16)")
                
            control_image = input_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"Control image size: {control_image.size} image {control_image}")
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            if status_callback:
                status_callback("Upscaling started, monitoring resources...")
            
            # Start memory monitoring with UI updates
            self.memory_monitor.start_monitoring(status_callback)
            
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
                
                # Apply CC effects to upscaled image if enabled
                if apply_cc_preset:
                    if status_callback:
                        status_callback("Applying color correction...")
                    output_image = self.apply_cc_effects(output_image)
                
                # Resize output image back to original image size multiplied by upscale factor
                original_upscaled_size = (w , h)
                print(f"Original image size: {original_upscaled_size}")
                print(f"Output image size before resize: {output_image.size}")
                output_image = output_image.resize(original_upscaled_size, Image.LANCZOS)
                
                return output_image, f"Upscaling completed successfully with seed: {seed}"
            
            finally:
                if status_callback:
                    status_callback("Cleaning up resources...")
                self.memory_monitor.stop_monitoring()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            error_msg = f"Error during upscaling: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

# Create upscaler instance and autoload model
upscaler = ImageUpscaler()

def auto_load_model():
    if not upscaler.model_loaded:
        status_text = "Loading model into memory..."
        
        success, message = upscaler.load_model(lambda msg: None)  # We'll handle updates differently
        
        if success:
            return f"✅ Model loaded successfully\n\n{message}"
        else:
            return f"❌ {message}"
    return "Model was already loaded"

def upscale_interface(
    input_image, 
    prompt, 
    guidance_scale, 
    apply_cc, 
    steps, 
    upscale_factor,
    controlnet_scale, 
    seed,
    progress=gr.Progress()
):
    if input_image is None:
        return None, "Please upload an image first."
    
    progress(0.1, desc="Starting upscaling")
    status_text = ""
    
    def status_callback(msg):
        nonlocal status_text
        # Append new message to existing status text
        status_text = status_text + "\n" + msg if status_text else msg
        # Extract progress if possible, otherwise use indeterminate progress
        if "inference step" in msg:
            try:
                current_step = int(msg.split("inference step")[0].strip().split()[-1])
                progress_value = current_step / steps
                progress(progress_value, desc=msg)
            except:
                progress(0.5, desc=msg)
        else:
            progress(0.5, desc=msg)
    
    # Convert seed to int or None
    if seed is not None and seed != "":
        try:
            seed = int(seed)
        except:
            seed = None
    else:
        seed = None
    
    result_image, message = upscaler.upscale_image(
        input_image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        apply_cc_preset=apply_cc,
        num_inference_steps=steps,
        upscale_factor=upscale_factor,
        controlnet_conditioning_scale=controlnet_scale,
        seed=seed,
        status_callback=status_callback
    )
    
    progress(1.0, desc="Completed")
    
    # Save the image with a unique filename
    if result_image:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/upscaled_{uuid.uuid4().hex[:8]}.jpg"
        result_image.save(filename)
        return result_image, f"{message}\n\nSaved to {filename}\n\nMonitoring Logs:\n{status_text}"
    else:
        return None, f"❌ {message}\n\nMonitoring Logs:\n{status_text}"

# Gradio UI layout
with gr.Blocks(title="FLUX.1 Image Upscaler") as demo:
    gr.Markdown("# FLUX.1 Image Upscaler")
    gr.Markdown("Upscale your images using FLUX.1 model")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="pil")
            with gr.Group():
                gr.Markdown("## Upscaling Options")
                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="Enter a prompt to guide upscaling, or leave empty",
                    lines=2
                )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=5.0,
                        step=0.1
                    )
                    controlnet_scale = gr.Slider(
                        label="ControlNet Scale",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.6,
                        step=0.05
                    )
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=8,
                        maximum=50,
                        value=28,
                        step=1
                    )
                    upscale_factor = gr.Slider(
                        label="Upscale Factor",
                        minimum=1,
                        maximum=4,
                        value=4,
                        step=1
                    )
                with gr.Row():
                    seed = gr.Textbox(
                        label="Seed (optional)",
                        placeholder="Leave empty for random seed"
                    )
                    apply_cc = gr.Checkbox(
                        label="Apply Color Correction",
                        value=False
                    )
            upscale_btn = gr.Button("Upscale Image", variant="primary")

        with gr.Column(scale=1):
            result_image = gr.Image(label="Result")
            status = gr.TextArea(
                label="Status",
                placeholder="Status will appear here...",
                interactive=False
            )
    
    # Automatically load model on startup
    demo.load(
        fn=lambda: auto_load_model(),
        outputs=None
    )
    
    # Event handler
    upscale_btn.click(
        fn=upscale_interface,
        inputs=[
            input_image,
            prompt,
            guidance_scale,
            apply_cc,
            steps,
            upscale_factor,
            controlnet_scale,
            seed
        ],
        outputs=[result_image, status]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production
