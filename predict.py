from typing import List
import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers.models import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image, ImageEnhance, ImageFilter
import warnings

class Predictor(BasePredictor):
    def apply_cc_effects(self, img: Image.Image) -> Image.Image:
        print("Applying CC effects")
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Adjust exposure (-1 in Lightroom scale ≈ -20% brightness)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.80)
        
        # Adjust contrast (+25 in Lightroom scale ≈ 25% increase)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)
        
        # Adjust vibrance (+16 in Lightroom scale ≈ subtle saturation boost)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.06)
        
        return img

    def setup(self) -> None:
        print("Loading the model into memory to make running multiple predictions efficient")
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
        apply_cc_preset: bool = Input(
            description="Apply CC Preset effect to image",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        print(f"Seed: {seed}")
        
        input_image = Image.open(input_image).convert("RGB")
        # Store original dimensions
        original_size = input_image.size
        
        input_image = self.process_input(input_image)

        # Prepare control image
        w, h = input_image.size
        control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Generating upscaled image")
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
            output_image = self.apply_cc_effects(output_image)

        # Resize back to original dimensions and save
        print("Saving output image")
        output_path = Path("output.jpg")
        output_image = output_image.resize(original_size, Image.LANCZOS)
        output_image.save(output_path)
        return output_path
