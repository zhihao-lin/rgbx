import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import gradio as gr
import torch
import torchvision
from PIL import Image
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
from argparse import ArgumentParser
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))

def load_intrinsics(
    albedo,
    normal,
    roughness,
    metallic,
    irradiance
):
    albedo_image = None
    if albedo.endswith(".exr"):
        albedo_image = load_exr_image(albedo, clamp=True).to("cuda")
    elif (
        albedo.endswith(".png")
        or albedo.endswith(".jpg")
        or albedo.endswith(".jpeg")
    ):
        albedo_image = load_ldr_image(albedo, from_srgb=True).to("cuda")

    normal_image = None
    if normal.endswith(".exr"):
        normal_image = load_exr_image(normal, normalize=True).to("cuda")
    elif (
        normal.endswith(".png")
        or normal.endswith(".jpg")
        or normal.endswith(".jpeg")
    ):
        normal_image = load_ldr_image(normal, normalize=True).to("cuda")
        
    roughness_image = None
    if roughness.endswith(".exr"):
        roughness_image = load_exr_image(roughness, clamp=True).to("cuda")
    elif (
        roughness.endswith(".png")
        or roughness.endswith(".jpg")
        or roughness.endswith(".jpeg")
    ):
        roughness_image = load_ldr_image(roughness, clamp=True).to("cuda")

    metallic_image = None
    if metallic.endswith(".exr"):
        metallic_image = load_exr_image(metallic, clamp=True).to("cuda")
    elif (
        metallic.endswith(".png")
        or metallic.endswith(".jpg")
        or metallic.endswith(".jpeg")
    ):
        metallic_image = load_ldr_image(metallic, clamp=True).to("cuda")

    irradiance_image = None
    if irradiance.endswith(".exr"):
        irradiance_image = load_exr_image(
            irradiance, tonemaping=True, clamp=True
        ).to("cuda")
    elif (
        irradiance.endswith(".png")
        or irradiance.endswith(".jpg")
        or irradiance.endswith(".jpeg")
    ):
        irradiance_image = load_ldr_image(
            irradiance, from_srgb=True, clamp=True
        ).to("cuda")

    # Set default height and width
    height = 768
    width = 768

    # Check if any of the input images are not None
    # and set the height and width accordingly
    images = [
        albedo_image,
        normal_image,
        roughness_image,
        metallic_image,
        irradiance_image,
    ]
    for img in images:
        if img is not None:
            height = img.shape[1]
            width = img.shape[2]
            break
    img_hw = (height, width)

    return images, img_hw

def main():
    parser = ArgumentParser()
    parser.add_argument("--dir_input", type=str, help="Input image path")
    parser.add_argument("--dir_output", type=str, help="Output image path")
    parser.add_argument("--inference_step", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--prompt", type=str, default="", help="Prompt")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="Image guidance scale")
    args = parser.parse_args()

    # Load pipeline
    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        "zheng95z/x-to-rgb",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    
    dir_input = args.dir_input
    dir_output = args.dir_output
    os.makedirs(dir_output, exist_ok=True)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    albedo_paths = sorted([os.path.join(dir_input, f) for f in os.listdir(dir_input) if "albedo" in f])
    normal_paths = sorted([os.path.join(dir_input, f) for f in os.listdir(dir_input) if "normal" in f])
    roughness_paths = sorted([os.path.join(dir_input, f) for f in os.listdir(dir_input) if "roughness" in f])
    metallic_paths = sorted([os.path.join(dir_input, f) for f in os.listdir(dir_input) if "metallic" in f])
    irradiance_paths = sorted([os.path.join(dir_input, f) for f in os.listdir(dir_input) if "irradiance" in f])

    n_frames = len(albedo_paths)
    for i in tqdm(range(n_frames)):
        intrinsic_images, img_hw = load_intrinsics(
            albedo_paths[i],
            normal_paths[i],
            roughness_paths[i],
            metallic_paths[i],
            irradiance_paths[i]
        )
        albedo_image, normal_image, roughness_image, metallic_image, irradiance_image = intrinsic_images
        height, width = img_hw
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        generated_image = pipe(
            prompt=args.prompt,
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=args.inference_step,
            height=height,
            width=width,
            generator=generator,
            required_aovs=required_aovs,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            guidance_rescale=0.7,
            output_type="np",
        ).images[0]

        save_path = os.path.join(dir_output, f"{os.path.basename(albedo_paths[i]).split('albedo')[0]}rgb.png")
        generated_image = (generated_image * 255).astype('uint8')
        generated_image = Image.fromarray(generated_image)
        generated_image.save(save_path)
        


if __name__ == "__main__":
    main()