import os
import torch
from diffusers import StableDiffusionPipeline

# ===== CONFIG =====
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

LORA_EXPERIMENTS = {
    "exp_e10": "output/exp_e10",
    "exp_e20": "output/exp_e20",
    "exp_e30": "output/exp_e30",
    "exp_e40": "output/exp_e40",
    "exp_e50": "output/exp_e50",
}

OUTPUT_ROOT = "generated_images"

IMAGES_PER_LORA = 10
WIDTH = 512
HEIGHT = 512  # portrait
STEPS = 35
GUIDANCE = 7.0
SEED_BASE = 1234  # keeps generations comparable
# ==================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Load base pipeline once
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

#pipe.enable_xformers_memory_efficient_attention()

prompt = (
    "portrait photo of <aravind-person>, same facial structure, same jawline, "
    "same eye spacing, same nose shape, realistic skin texture, "
    "DSLR photo, 85mm lens, natural lighting"
)

negative_prompt = (
    "cartoon, anime, illustration, painting, blurry, "
    "distorted face, extra fingers, bad anatomy, bad eyes"
)

for exp_name, lora_path in LORA_EXPERIMENTS.items():
    print(f"\n🚀 Generating images for {exp_name}")

    if not os.path.exists(lora_path):
        print(f"❌ LoRA not found: {lora_path}")
        continue

    # Output folder per experiment
    exp_out = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(exp_out, exist_ok=True)

    # Load LoRA
    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_path)

    for i in range(IMAGES_PER_LORA):
        seed = SEED_BASE + i
        generator = torch.Generator("cuda").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator
        ).images[0]

        filename = f"img_{i+1:02d}_seed{seed}.png"
        save_path = os.path.join(exp_out, filename)
        image.save(save_path)

        print(f"✅ {exp_name} | saved {filename}")

print("\n🎉 All LoRA experiments generated successfully!")
