import torch
import argparse
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetImg2ImgPipeline
from torch import autocast
from math import floor
from random import choice, random
from os import listdir

parser = argparse.ArgumentParser(description='Generate a metal band logo')

parser.add_argument('prompt', type=str, help='Prompt to use when generating the final image')
parser.add_argument('base_prompt', type=str, help='Prompt to use when generating the base image')
parser.add_argument('--model-dir', type=str, required=True, help='Path to Stable Diffusion model directory')
parser.add_argument('--band-name', type=str, help='Text to use for band name')
parser.add_argument('--strength', type=float, default=1.5, help='ControlNet strength')
parser.add_argument('--steps', type=int, default=20, help='Number of steps per image')
parser.add_argument('--width', type=int, default=640, help='Image width')
parser.add_argument('--height', type=int, default=360, help='Image height')

def read_directory(path):
    lines = listdir(path)
    lines = [f'{path}/{line}' for line in lines]
    return lines

def read_prompt_file(name):
    print(f'Reading {name}.txt')
    with open(f'./{name}.txt', 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    print(f'Loaded {len(lines)} items')
    return lines

def pick_one(arr):
    return choice(arr).strip().replace('\n', '')

args = parser.parse_args()

if not args.band_name:
    band_names = read_prompt_file('bands')
    args.band_name = pick_one(band_names)

font_path = pick_one(read_directory('./fonts'))

# create controlnet input

image_outer = Image.new("RGBA", (args.width, args.height), (0,0,0,255))
image = Image.new("RGBA", (floor(args.width / 1.2), floor(args.height / 1.2)), (0,0,0,255))
draw = ImageDraw.Draw(image)

font_size = 128
font = ImageFont.truetype(font_path, font_size)
text_width, text_height = draw.textsize(args.band_name, font=font)

while text_width >= image.width:
    font_size -= 8
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(args.band_name, font=font)

print(f'Best font size is {font_size}')

draw.text((floor(image.width / 2), floor(image.height / 2)), args.band_name, (255,255,255), anchor="mm", font=font)
image_outer.paste(image, (floor((image_outer.width - image.width) / 2), floor((image_outer.height - image.height) / 2)))
image_outer.save('./controlnet-input.png')

# proceed with image generation

device = torch.device('cuda')
scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule='scaled_linear'
)

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_dir,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to(device)
image_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    args.model_dir,
    scheduler=scheduler,
    controlnet=controlnet,
    safety_checker=None
)
image_pipe = image_pipe.to(device)

seed = floor(random() * 1000000000)
generator = torch.Generator('cuda').manual_seed(seed)

# run iterations and save output
with autocast("cuda"):
    with torch.inference_mode():
        base_image = pipe(
            prompt=args.base_prompt,
            num_inference_steps=args.steps,
            generator=generator,
            height=args.height,
            width=args.width
        ).images[0]
        base_image.save('./base.png')

        image = image_pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            generator=generator,
            height=args.height,
            width=args.width,
            strength=0.6,
            controlnet_conditioning_scale=args.strength,
            image=base_image,
            control_image=image_outer
        ).images[0]
        image.save('./composite.png')