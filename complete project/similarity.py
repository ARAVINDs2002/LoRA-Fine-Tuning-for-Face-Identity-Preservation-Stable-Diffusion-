import os
import numpy as np
from deepface import DeepFace

# ===== PATHS =====
REAL_DIR = "lora_dataset/aravind"  # Update 'aravind' to your subject name
GEN_DIR  = "generated_images"

EXPS = ["exp_e10", "exp_e20", "exp_e30", "exp_e40", "exp_e50"]

MODEL_NAME = "ArcFace"    # industry standard
METRIC = "cosine"         # cosine similarity

# ==================

# Load real images
real_images = [
    os.path.join(REAL_DIR, f)
    for f in os.listdir(REAL_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print("\n🔍 FACE SIMILARITY EVALUATION (ArcFace, mean of 10 images)\n")

for exp in EXPS:
    exp_path = os.path.join(GEN_DIR, exp)

    if not os.path.isdir(exp_path):
        print(f"⚠️ {exp} → folder not found")
        continue

    gen_images = [
        os.path.join(exp_path, f)
        for f in os.listdir(exp_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(gen_images) == 0:
        print(f"⚠️ {exp} → no generated images")
        continue

    gen_level_scores = []  # one score per generated image

    for gen_img in gen_images:
        similarities = []

        for real_img in real_images:
            try:
                result = DeepFace.verify(
                    img1_path=gen_img,
                    img2_path=real_img,
                    model_name=MODEL_NAME,
                    distance_metric=METRIC,
                    enforce_detection=True
                )

                similarity = 1 - result["distance"]
                similarities.append(similarity)

            except:
                continue

        if similarities:
            # take BEST match for this generated image
            gen_level_scores.append(max(similarities))

    if gen_level_scores:
        avg_score = np.mean(gen_level_scores)
        min_score = np.min(gen_level_scores)
        max_score = np.max(gen_level_scores)

        print(
            f"📊 {exp} → "
            f"Avg: {avg_score:.3f} | "
            f"Min: {min_score:.3f} | "
            f"Max: {max_score:.3f}"
        )
    else:
        print(f"⚠️ {exp} → no valid face matches")

print("\n✅ Evaluation completed\n")
