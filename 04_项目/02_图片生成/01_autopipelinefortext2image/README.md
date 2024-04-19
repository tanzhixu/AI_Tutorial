### 项目地址
huggingface.co/stabilityai/sdxl-turbo

### 安装Diffusers
```
pip install diffusers transformers accelerate --upgrade
```

### 提示词网址
https://mspoweruser.com/best-stable-diffusion-prompts

### Text-to-image
SDXL-Turbo does not make use of guidance_scale or negative_prompt, we disable it with guidance_scale=0.0. Preferably, the model generates images of size 512x512 but higher image sizes work as well. A single step is enough to generate high quality images.

```
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

```

### Image-to-image
When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1. The image-to-image pipeline will run for int(num_inference_steps * strength) steps, e.g. 0.5 * 2.0 = 1 step in our example below.

```
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

init_image = load_image("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]

```