import os
import torch
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel, FluxControlNetModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageEnhance
### not used here, but useful for debuggingS
# import numpy as np
# import threading
# import time
# import psutil
# import pynvml
import gc
import argparse
from pathlib import Path
from huggingface_hub import login, snapshot_download

# Memory and performance optimizations
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["ACCELERATE_USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"  # Using bfloat16 for better memory efficiency

def flush():
    """Thoroughly clean GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

class ImageUpscaler:
    def __init__(self):
        self.hf_token = "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"
        self.max_memory = {
            0: "11GB",  # Leave 1GB buffer on each GPU
            1: "11GB",
            2: "11GB",
            3: "11GB",
            4: "11GB",
            5: "11GB"
        }
        self.setup()

    def print_gpu_memory(self):
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            used = torch.cuda.memory_allocated(i) / 1e9
            print(f"GPU {i}: Using {used:.1f}GB / {total:.1f}GB")

    def setup(self):
        print("\nLoading models in stages...")
        flush()

        # Stage 1: Load and generate text embeddings
        print("Stage 1: Loading text encoders and generating embeddings...")
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=None,
            vae=None,
            device_map="balanced",
            max_memory=self.max_memory,
            torch_dtype=torch.bfloat16
        )
        
        # Generate base embeddings for upscaling
        with torch.no_grad():
            self.prompt_embeds, self.pooled_prompt_embeds, _ = pipeline.encode_prompt(
                prompt="enhance image quality, sharpen details",
                prompt_2=None,
                max_sequence_length=512
            )

        # Clean up text encoders
        del pipeline.text_encoder
        del pipeline.text_encoder_2
        del pipeline.tokenizer
        del pipeline.tokenizer_2
        del pipeline
        flush()

        # Stage 2: Load transformer for processing
        print("Stage 2: Loading transformer model...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            device_map="balanced",
            max_memory=self.max_memory,
            torch_dtype=torch.bfloat16
        )

        # Stage 3: Load ControlNet
        print("Stage 3: Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="balanced",
            max_memory=self.max_memory
        )

        # Stage 4: Initialize pipeline with loaded components
        print("Stage 4: Initializing pipeline...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            vae=None,
            transformer=self.transformer,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16
        )

        # Stage 5: Load VAE on first GPU
        print("Stage 5: Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to("cuda:0")  # VAE on first GPU
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        print("\nModel loading complete. Current GPU memory usage:")
        self.print_gpu_memory()

    def process_input(self, input_image: Image.Image, max_size: int = 512) -> Image.Image:
        w, h = input_image.size
        aspect_ratio = h / w
        
        if w > h:
            new_w = min(w, max_size)
            new_h = int(new_w * aspect_ratio)
        else:
            new_h = min(h, max_size)
            new_w = int(new_h / aspect_ratio)
            
        # Ensure dimensions are divisible by 8
        new_w = new_w - new_w % 8
        new_h = new_h - new_h % 8
        
        return input_image.resize((new_w, new_h), Image.LANCZOS)

    def upscale(self, input_path: str, output_path: str = "output.jpg", prompt: str = "",
                guidance_scale: float = 3.5, apply_cc_preset: bool = False,
                num_inference_steps: int = 20, upscale_factor: float = 2.0,
                controlnet_conditioning_scale: float = 0.6, seed: int = None) -> str:
        
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,))[0].item())
        
        print(f"Processing with seed: {seed}")
        
        # Load and prepare input image
        input_image = Image.open(input_path).convert("RGB")
        original_size = input_image.size
        input_image = self.process_input(input_image)
        
        # Calculate target dimensions
        w, h = input_image.size
        target_w = int(w * upscale_factor)
        target_h = int(h * upscale_factor)
        target_w = target_w - target_w % 8
        target_h = target_h - target_h % 8
        
        print(f"Processing image from {w}x{h} to {target_w}x{target_h}")
        
        # Prepare control image
        control_image = input_image.resize((target_w, target_h))
        
        try:
            # Run denoising
            print("Running denoising...")
            latents = self.pipe(
                prompt_embeds=self.prompt_embeds,
                pooled_prompt_embeds=self.pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=target_h,
                width=target_w,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                output_type="latent"
            ).images
            
            # Clean up GPU memory
            flush()
            
            # Decode latents
            print("Decoding latents...")
            with torch.no_grad():
                latents = self.pipe._unpack_latents(latents, target_h, target_w, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                output_image = self.image_processor.postprocess(image, output_type="pil")[0]
            
            # Apply color correction if requested
            if apply_cc_preset:
                output_image = ImageEnhance.Brightness(output_image).enhance(0.60)
                output_image = ImageEnhance.Contrast(output_image).enhance(1.25)
                output_image = ImageEnhance.Color(output_image).enhance(1.06)
            
            # Save result
            print("Saving output image...")
            output_path = Path(output_path)
            output_image = output_image.resize(
                (int(original_size[0] * upscale_factor), int(original_size[1] * upscale_factor)),
                Image.LANCZOS
            )
            output_image.save(output_path)
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
        finally:
            # Clean up
            flush()
            self.print_gpu_memory()

def main():
    parser = argparse.ArgumentParser(description="Image Upscaler using FLUX.1")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--output", default="output.jpg", help="Path for the output image")
    parser.add_argument("--prompt", default="", help="Optional text prompt to guide upscaling")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale (1.0-20.0)")
    parser.add_argument("--apply-cc", action="store_true", help="Apply color correction")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--upscale-factor", type=float, default=2.0, help="Upscale factor")
    parser.add_argument("--controlnet-scale", type=float, default=0.6, help="ControlNet conditioning scale")
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