import gradio as gr
import torch
from PIL import Image
import gc
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image, ImageEnhance
from huggingface_hub import login, snapshot_download

def apply_cc_effects(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.60)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.25)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.06)
    
    return img

def upscale_image(
    input_image,
    prompt,
    guidance_scale,
    apply_cc_preset,
    num_inference_steps,
    upscale_factor,
    controlnet_conditioning_scale,
    seed
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if seed is None or seed == 0:
        seed = int(torch.randint(0, 1000000, (1,))[0].item())

    # Load models if not already loaded
    if not hasattr(upscale_image, 'pipe'):
        hf_token = "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"
        login(token=hf_token)
        
        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            repo_type="model",
            ignore_patterns=["*.md", "*.gitattributes"],
            local_dir="FLUX.1-dev",
            token=hf_token,
        )
        
        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        ).to(device)
        
        upscale_image.pipe = FluxControlNetPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16 
        ).to(device)

    # Prepare control image
    w, h = input_image.size
    control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate image
    output_image = upscale_image.pipe(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
    ).images[0]

    # Apply CC effects if enabled
    if apply_cc_preset:
        output_image = apply_cc_effects(output_image)

    # Clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_image

# Create Gradio interface
demo = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Prompt", placeholder="Enter prompt or leave empty"),
        gr.Slider(1.0, 20.0, value=5.0, label="Guidance Scale"),
        gr.Checkbox(label="Apply Color Correction"),
        gr.Slider(8, 50, value=28, step=1, label="Number of Inference Steps"),
        gr.Slider(1, 4, value=4, step=1, label="Upscale Factor"),
        gr.Slider(0.1, 1.5, value=0.6, label="Controlnet Conditioning Scale"),
        gr.Number(value=0, label="Seed (0 for random)", precision=0),
    ],
    outputs=gr.Image(type="pil", label="Upscaled Image"),
    title="FLUX.1 Image Upscaler",
    description="Upload an image to upscale it using FLUX.1. Optionally provide a prompt to guide the upscaling process."
)

if __name__ == "__main__":
    demo.launch()
