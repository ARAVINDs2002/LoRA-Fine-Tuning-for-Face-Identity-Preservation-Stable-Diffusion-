# LoRA Fine-Tuning for Face Identity Preservation (Stable Diffusion)

## Overview

This project focuses on my first try on fine-tuning a Stable Diffusion model using **LoRA** to generate images of a **specific real person** (myself), while preserving facial identity as much as possible.

The goal was not just image generation, but to **measure identity preservation using numeric evaluation**, instead of relying only on visual inspection.

---

## Base Model

- **Model:** Stable Diffusion v1.5  
- **Source:** runwayml/stable-diffusion-v1-5  
- **Framework:** Hugging Face Diffusers  
- **Hardware:** RTX 3050 (6GB VRAM)

---

## Dataset

- Around **30 real images** of the same person
- Realistic photos with:
  - Natural lighting
  - Different angles
  - Different facial expressions
- Images stored in a single folder (no class labels)

**Example structure:**

```
lora_dataset/aravind/
├── img001.jpg
├── img002.jpg
├── img003.jpg
└── ...
```

---

## LoRA Fine-Tuning Setup

- **Training method:** LoRA (Low-Rank Adaptation)
- **Resolution:** 512 × 512
- **Batch size:** 1 (GPU memory limitation)
- **Rank:** 8
- **Mixed precision:** FP16
- **Optimizer:** Default Diffusers setup
- **Seed:** Fixed for reproducibility

---

## Experiments (Epoch-wise Training)

The same dataset was trained multiple times using different epoch values:

- **exp_e10** → 10 epochs  
- **exp_e20** → 20 epochs  
- **exp_e30** → 30 epochs  
- **exp_e40** → 40 epochs  
- **exp_e50** → 50 epochs  

Each experiment produced a separate LoRA weights file (`.safetensors`).

---

## Image Generation

For each trained LoRA:

- Images were generated using the **same prompt**
- **Same seed (1236)** was used across all experiments for fair comparison
- Portrait-style images were generated (512 × 768)

**Directory structure:**

```
generated_images/
├── exp_e10/
├── exp_e20/
├── exp_e30/
├── exp_e40/
└── exp_e50/
```

Only **one representative image per experiment** is shared below for visual reference.  
Batch image outputs are not included — only the final numeric evaluation results.

---
## Generated Image Results

All images below were generated using the **same prompt and the same seed (1236)**  
This ensures a fair comparison across different training epochs.

<table>
  <tr>
    <th>10 Epochs</th>
    <th>20 Epochs</th>
    <th>30 Epochs</th>
    <th>40 Epochs</th>
    <th>50 Epochs</th>
  </tr>
  <tr>
    <td align="center">
      <img src="exp_e10/img_03_seed1236.png" width="150"/>
    </td>
    <td align="center">
      <img src="exp_e20/img_03_seed1236.png" width="150"/>
    </td>
    <td align="center">
      <img src="exp_e30/img_03_seed1236.png" width="150"/>
    </td>
    <td align="center">
      <img src="exp_e40/img_03_seed1236.png" width="150"/>
    </td>
    <td align="center">
      <img src="exp_e50/img_03_seed1236.png" width="150"/>
    </td>
  </tr>
</table>

Only **one representative image per experiment** is shown here for visual comparison.  
Batch outputs are evaluated numerically and reported in the results table below.

## Training Data (Sample Images)

Below are a few **sample images from the real dataset** used to fine-tune the LoRA.  
These images show variations in lighting, angle, and expression.

<table>
  <tr>
    <td align="center">
      <img src="sample_images/photo_2026-01-31_13-12-23.jpg" width="120"/>
    </td>
    <td align="center">
      <img src="sample_images/photo_2026-01-31_13-12-25.jpg" width="120"/>
    </td>
    <td align="center">
      <img src="sample_images/photo_2026-01-31_13-12-27.jpg" width="120"/>
    </td>
    <td align="center">
      <img src="sample_images/photo_2026-01-31_13-12-28.jpg" width="120"/>
    </td>
    <td align="center">
      <img src="sample_images/photo_2026-01-31_13-12-29.jpg" width="120"/>
    </td>
  </tr>
</table>

These images are shared only to illustrate the type of data used for training.

---

## Prompt Configuration

The following prompt and negative prompt were used for all image generations:

```python
# Prompt used for all image generations
# This prompt is designed to preserve my facial identity
# by emphasizing consistent facial structure and realistic photography traits.

prompt = (
    "portrait photo of <aravind-person>, "
    "same facial structure, same jawline, same eye spacing, same nose shape, "
    "realistic skin texture, "
    "DSLR photo, 85mm lens, natural lighting"
)

# Negative prompt used to avoid common generation issues
# This helps reduce distortions, unrealistic styles, and anatomy errors.

negative_prompt = (
    "cartoon, anime, illustration, painting, blurry, "
    "distorted face, extra fingers, bad anatomy, bad eyes"
)
```

---

## Why Numeric Evaluation Was Used

Visual inspection alone is subjective.

Although Stable Diffusion operates in latent space, **identity is not explicitly disentangled** there.  
Because of this, direct latent comparison is not reliable for identity evaluation.

To address this, a **face recognition embedding model** was used to provide objective, numeric results.

---

## Face Similarity Evaluation (Primary Method)

### Model Used

- **ArcFace**
- Widely used in face verification and identity recognition systems

### Metric

- **Cosine similarity**  
- **Range:** 0 to 1  
  - Higher value = more similar facial identity

---

## Batch Image Face Similarity (Main Evaluation)

To reduce randomness from generation seeds, **batch-based evaluation** was used as the primary metric.

**Process:**

1. Generate **10 images per experiment**
2. For each generated image:
   - Compare it with all real images
   - Keep the best similarity score
3. Aggregate results across the batch:
   - Average similarity
   - Minimum similarity
   - Maximum similarity

This approach provides a more stable and realistic measurement of identity preservation.

---

## Batch Face Similarity Results

| Experiment | Avg Similarity | Min   | Max   |
|------------|----------------|-------|-------|
| exp_e10    | 0.459          | 0.394 | 0.551 |
| exp_e20    | 0.466          | 0.360 | 0.592 |
| exp_e30    | 0.421          | 0.194 | 0.569 |
| exp_e40    | 0.510          | 0.223 | 0.978 |
| exp_e50    | **0.573**      | 0.373 | 0.703 |

The strongest identity preservation was observed between **30 and 50 epochs**.

---

## Observations

- Too few epochs → identity not learned sufficiently
- Too many epochs → overfitting or facial distortion
- Certain seeds produce significantly better identity matches
- LoRA learns identity patterns, not exact face replication
- Generative models may hallucinate due to:
  - Random sampling
  - Prompt influence
  - Limited training data

---

## Conclusion

- LoRA fine-tuning can partially preserve facial identity
- Batch-based ArcFace cosine similarity provides a reliable numeric evaluation
- Best results were achieved around **30–50 epochs**
- Numeric evaluation combined with visual inspection gives the clearest understanding

---

## Notes

- This project is for research and learning purposes only
- Generated images are not intended for impersonation or misuse
- Evaluation methods follow commonly accepted industry practices

---

## License

This project is for educational and research purposes only.


