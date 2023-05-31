import os
from typing import List

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

from PIL import Image

from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
#Import the face detection library
import cv2

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "Nurialeb207/sd_face"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="PHOTOREALISTIC FULL BODY PICTURE OF AIARA. Aiara is a thin sexy woman with mid size boobs, a big"
                    " sexy ass and a remarkable waist hip ratio. The abdominal muscles in the belly area are marked."
                    " She is a real person with a realistic face and realistic body with natural body marks.She is"
                    " wearing a pink sexy corset with matching thong and stockings. She is standing in a bathroom only "
                    "fans picture wide shot portrait, sharp, photography, Nikon D850, 50mm, f/2.8",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="fat,out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, "
                    "duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands,"
                    " poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, "
                    "extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, "
                    "missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, "
                    "watermark, signature,, asymmetrical eyes, background crowd"
        ),
        image: Path = Input(
            description="Inital image to generate variations of. Supproting images size with 512x512",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        print(image)
        image_name=image
        image = Image.open(image).convert("RGB").resize((512, 512))
        #Load the image as cv2
        image_cv2 = cv2.imread(image_name)
        if mask is None:
            mask=make_face_swap(image_cv2)
        else:
            mask = Image.open(mask).convert("RGB").resize(image.size)
        extra_kwargs = {
            "mask_image": mask,
            "image": image
        }

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

def make_face_swap(picture_path):
    #Step 1: Find the faces in the picture
    img = cv2.imread(picture_path)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        # Enhace the face area on a 30% to make sure it includes hair, etc
        x = int(x - 0.3 * w)
        y = int(y - 0.3 * h)
        w = int(w * 1.6)
        h = int(h * 1.6)
        face_img = img[y: y + h, x: x + w]
        face = (x, y, w, h)
    #Generate a black and white mask of the face
    mask = np.zeros(img.shape, np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    # Convert the mask into a PIL image
    mask = Image.fromarray(mask)
    return mask
def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
