import gradio as gr
import threading
import time
import os
import uuid
from PIL import Image, ImageEnhance
import random

class MockUpscaler:
    def __init__(self):
        self.keep_running = False
        self.monitor_thread = None

    def simulate_system_info(self, status_callback=None):
        if status_callback:
            status_callback("Mock System Info:\nCPU: Intel i9-13900K (Mock)\nGPU: NVIDIA RTX 4090 24GB (Mock)")

    def monitor_resources(self, status_callback=None):
        start_time = time.time()
        while self.keep_running:
            # Simulate increasing resource usage over time
            elapsed = time.time() - start_time
            gpu_usage = min(95, 30 + (elapsed * 2))  # Gradually increase GPU usage
            ram_usage = min(85, 20 + (elapsed * 1.5))  # Gradually increase RAM usage
            
            status_msg = (
                f"[Mock Stats]\n"
                f"GPU: {gpu_usage:.1f}% | Memory: 14.2/24.0 GB\n"
                f"RAM: {ram_usage:.1f}% | CPU: {random.randint(20, 60)}%"
            )
            
            if status_callback:
                status_callback(status_msg)
            
            time.sleep(3)

    def simulate_processing(self, total_steps: int, status_callback):
        for step in range(total_steps):
            time.sleep(30 / total_steps)  # Distribute 30 seconds across steps
            if status_callback:
                status_callback(f"Processing step {step + 1}/{total_steps}")

    def apply_mock_upscale(self, img: Image.Image, upscale_factor: int) -> Image.Image:
        # Simple resize operation
        w, h = img.size
        return img.resize((w * upscale_factor, h * upscale_factor), Image.Resampling.LANCZOS)

    def apply_mock_effects(self, img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.95)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        return img

    def upscale_image(
        self,
        input_image,
        prompt="",
        guidance_scale=5.0,
        apply_cc=False,
        num_inference_steps=28,
        upscale_factor=4,
        controlnet_scale=0.6,
        seed=None,
        status_callback=None,
    ):
        if input_image is None:
            return None, "No input image provided."
        
        try:
            # Generate seed if not provided
            if seed is None or seed == "":
                seed = random.randint(0, 1000000)
            
            self.simulate_system_info(status_callback)
            
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            
            input_image = input_image.convert("RGB")
            
            if status_callback:
                status_callback("Starting mock upscaling process...")
            
            # Start resource monitoring
            self.keep_running = True
            self.monitor_thread = threading.Thread(target=self.monitor_resources, args=(status_callback,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            try:
                # Simulate processing steps
                self.simulate_processing(num_inference_steps, status_callback)
                
                # Apply mock upscaling
                output_image = self.apply_mock_upscale(input_image, upscale_factor)
                
                # Apply mock CC effects if enabled
                if apply_cc:
                    if status_callback:
                        status_callback("Applying color correction...")
                    output_image = self.apply_mock_effects(output_image)
                
                return output_image, f"Mock upscaling completed with seed: {seed}"
            
            finally:
                if status_callback:
                    status_callback("Cleaning up...")
                self.keep_running = False
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=1.0)
        
        except Exception as e:
            error_msg = f"Error during mock upscaling: {str(e)}"
            return None, error_msg

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
    
    progress(0.1, desc="Starting mock upscaling")
    status_text = ""
    
    def status_callback(msg):
        nonlocal status_text
        status_text = msg
        if "step" in msg:
            try:
                current_step = int(msg.split("step")[1].split("/")[0])
                total_steps = int(msg.split("/")[1])
                progress(current_step / total_steps, desc=msg)
            except:
                progress(0.5, desc=msg)
        else:
            progress(0.5, desc=msg)
    
    # Initialize upscaler
    upscaler = MockUpscaler()
    
    result_image, message = upscaler.upscale_image(
        input_image=input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        apply_cc=apply_cc,
        num_inference_steps=steps,
        upscale_factor=upscale_factor,
        controlnet_scale=controlnet_scale,
        seed=seed,
        status_callback=status_callback
    )
    
    progress(1.0, desc="Completed")
    
    if result_image:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/mock_upscaled_{uuid.uuid4().hex[:8]}.jpg"
        result_image.save(filename)
        return result_image, f"{message}\n\nSaved to {filename}\n\n{status_text}"
    else:
        return None, f"❌ {message}\n\n{status_text}"

# Gradio UI layout
with gr.Blocks(title="Mock FLUX.1 Image Upscaler") as demo:
    gr.Markdown("# Mock FLUX.1 Image Upscaler")
    gr.Markdown("Test interface with 30-second mock processing")
    
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

if __name__ == "__main__":
    demo.launch(share=False)
