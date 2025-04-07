from typing import Optional, Tuple
import torch
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image
import threading
import time
import psutil
import gc
import pynvml
import os
import gradio as gr
import uuid
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Maximum pixel budget for the upscaled image
MAX_PIXEL_BUDGET = 2048 * 2048  # Can be adjusted based on GPU memory

def handle_dimensions(input_image: Image.Image, upscale_factor: int) -> Tuple[Image.Image, int, int, bool]:
    """
    Handle image dimensions while maintaining aspect ratio and checking against MAX_PIXEL_BUDGET.
    Returns: (processed_image, target_width, target_height, was_resized)
    """
    w, h = input_image.size
    aspect_ratio = w / h

    # Calculate target dimensions
    target_w = w * upscale_factor
    target_h = h * upscale_factor
    was_resized = False

    # Check if target size exceeds pixel budget
    if target_w * target_h > MAX_PIXEL_BUDGET:
        warnings.warn(
            f"Requested output image is too large ({target_w}x{target_h}). Adjusting dimensions to fit pixel budget."
        )
        gr.Info(
            f"Requested output image is too large ({target_w}x{target_h}). Adjusting dimensions to fit pixel budget."
        )
        # Calculate new target dimensions that fit within budget
        scale = (MAX_PIXEL_BUDGET / (target_w * target_h)) ** 0.5
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
        was_resized = True

    # Ensure dimensions are multiples of 16
    target_w = target_w - (target_w % 16)
    target_h = target_h - (target_h % 16)

    # Ensure input is multiple of 16 as well
    w = w - (w % 16)
    h = h - (h % 16)
    input_image = input_image.resize((w, h))

    return input_image, target_w, target_h, was_resized

class MemoryMonitor:
    def __init__(self):
        self.keep_running = False
        self.thread = None
        self.memory_stats = {}
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

    def monitor_memory(self, status_callback=None):
        while self.keep_running:
            stats = self.get_memory_stats()
            if torch.cuda.is_available():
                vram_info = f"VRAM: {stats['gpu_used']:.1f}GB/{stats['gpu_total']:.1f}GB"
                gpu_info = f"GPU Usage: {stats['gpu_util']}%"
            else:
                vram_info = "VRAM: N/A"
                gpu_info = "GPU: N/A"
            
            ram_info = f"RAM: {stats['ram_used']:.1f}GB/{stats['ram_total']:.1f}GB"
            cpu_info = f"CPU: {stats['cpu_percent']}%"
            
            status_msg = f"{vram_info}, {gpu_info}, {ram_info}, {cpu_info}"
            logger.info(status_msg)
            
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
    
    def load_model(self, status_callback=None) -> Tuple[bool, str]:
        if status_callback:
            status_callback("Loading model into memory...")
        
        try:
            from huggingface_hub import login, snapshot_download
            
            hf_token = "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"
            if hf_token:
                login(token=hf_token)
                if status_callback:
                    status_callback("HF token authenticated")

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
        input_image: Image.Image,
        prompt: str = "",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        upscale_factor: int = 4,
        controlnet_conditioning_scale: float = 0.6,
        seed: Optional[int] = None,
        status_callback=None,
    ) -> Tuple[Optional[Image.Image], str]:
        if not self.model_loaded:
            return None, "Model not loaded. Please load the model first."
        
        if input_image is None:
            return None, "No input image provided."
        
        try:
            # Generate seed if not provided
            if seed is None:
                seed = int(torch.randint(0, 1000000, (1,))[0].item())
            
            if status_callback:
                status_callback(f"Using seed: {seed}")
            
            # Ensure input is RGB
            input_image = input_image.convert("RGB")
            
            # Store original size
            original_size = input_image.size
            
            # Handle dimensions
            input_image, target_width, target_height, was_resized = handle_dimensions(input_image, upscale_factor)
            
            if was_resized:
                logger.info(f"Target dimensions adjusted to {target_width}x{target_height}")
            
            if status_callback:
                status_callback("Starting upscaling process...")
            
            # Start memory monitoring
            self.memory_monitor.start_monitoring(status_callback)
            
            try:
                # Generate upscaled image
                output_image = self.pipe(
                    prompt=prompt,
                    control_image=input_image,  # Use original size image
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=target_height,  # Specify target height
                    width=target_width,    # Specify target width
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                ).images[0]
                
                # Resize to original target size if it was adjusted
                if was_resized:
                    target_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
                    output_image = output_image.resize(target_size, Image.Resampling.LANCZOS)
                
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

def save_jpg_image(image: Image.Image, output_dir: str, original_size: tuple) -> str:
    """Save image as JPG with quality=95 after resizing to original dimensions"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/upscaled_{uuid.uuid4().hex[:8]}.jpg"
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize back to original dimensions
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    
    # Save with high quality JPG settings
    image.save(filename, format='JPEG', quality=95, optimize=True)
    return filename

def get_original_size(image: Image.Image) -> tuple:
    """Get original dimensions before any processing"""
    return image.size

def upscale_interface(
    input_image, 
    prompt: str,
    guidance_scale: float,
    steps: int,
    upscale_factor: int,
    controlnet_scale: float,
    seed: Optional[str],
    progress=gr.Progress()
):
    if input_image is None:
        return None, "Please upload an image first."
    
    upscaler = ImageUpscaler()
    status_text = ""
    
    def status_callback(msg):
        nonlocal status_text
        status_text = msg
        if "inference step" in msg:
            try:
                current_step = int(msg.split("inference step")[0].strip().split()[-1])
                progress(current_step / steps, desc=msg)
            except:
                progress(0.5, desc=msg)
        else:
            progress(0.5, desc=msg)
    
    progress(0.1, desc="Loading model...")
    success, message = upscaler.load_model(status_callback)
    if not success:
        return None, f"Failed to load model: {message}"
    
    # Convert seed to int or None
    if seed and seed.strip():
        try:
            seed_int = int(seed)
        except ValueError:
            return None, "Invalid seed value. Please enter a valid integer."
    else:
        seed_int = None
    
    result_image, message = upscaler.upscale_image(
        input_image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        upscale_factor=upscale_factor,
        controlnet_conditioning_scale=controlnet_scale,
        seed=seed_int,
        status_callback=status_callback
    )
    
    progress(1.0, desc="Completed")
    
    if result_image:
        # Store original size at the start
        original_size = get_original_size(input_image)
        filename = save_jpg_image(result_image, "outputs", original_size)
        return result_image, f"{message}\n\nSaved to {filename}\n\n{status_text}"
    else:
        return None, f"❌ {message}\n\n{status_text}"

# Gradio interface
with gr.Blocks(title="FLUX Image Upscaler v7.5") as demo:
    gr.Markdown("# FLUX Image Upscaler v7.5")
    gr.Markdown("Memory-efficient upscaling with JPG output format")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="pil")
            with gr.Group():
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
                seed = gr.Textbox(
                    label="Seed (optional)",
                    placeholder="Leave empty for random seed"
                )
            upscale_btn = gr.Button("Upscale Image", variant="primary")

        with gr.Column(scale=1):
            result_image = gr.Image(label="Result", format="jpg")  # Explicitly set format to jpg
            status = gr.TextArea(
                label="Status",
                placeholder="Status will appear here...",
                interactive=False
            )
    
    # Event handler
    upscale_btn.click(
        fn=upscale_interface,
        inputs=[
            input_image,
            prompt,
            guidance_scale,
            steps,
            upscale_factor,
            controlnet_scale,
            seed
        ],
        outputs=[result_image, status]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=False in production
