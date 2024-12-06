import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import gradio as gr
import torch
import torchvision
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from argparse import ArgumentParser
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))

def load_image(image_path):
    image = None
    if image_path.endswith(".exr"):
        image = load_exr_image(image_path, tonemaping=True, clamp=True).to("cuda")
    elif (
        image_path.endswith(".png")
        or image_path.endswith(".jpg")
        or image_path.endswith(".jpeg")
    ):
        image = load_ldr_image(image_path, from_srgb=True).to("cuda")

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height = image.shape[1]
    old_width = image.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000

    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)
    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8
    
    image = torchvision.transforms.Resize((new_height, new_width))(image)
    shape_old = (old_height, old_width)
    shape_new = (new_height, new_width)

    return image, shape_old, shape_new

def main():
    parser = ArgumentParser()
    parser.add_argument("--dir_input", type=str, help="Input image path")
    parser.add_argument("--dir_output", type=str, help="Output image path")
    parser.add_argument("--inference_step", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Load pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    dir_input = args.dir_input
    dir_output = args.dir_output
    os.makedirs(dir_output, exist_ok=True)
    image_names = sorted([name for name in os.listdir(dir_input)])

    for image_name in tqdm(image_names):
        image_path = os.path.join(dir_input, image_name)
        image, shape_old, shape_new = load_image(image_path)
        for aov_name in required_aovs:
            
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            prompt = prompts[aov_name]
            generated_image = pipe(
                prompt=prompt,
                photo=image,
                num_inference_steps=args.inference_step,
                height=shape_new[0],
                width=shape_new[1],
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            generated_image = torchvision.transforms.Resize(
                shape_old,
            )(generated_image)

            path = os.path.join(dir_output, f"{image_name.split('.')[0]}_{aov_name}.png")
            generated_image.save(path)



if __name__ == "__main__":
    main()