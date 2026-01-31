import os
import subprocess
import sys

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = "lora_dataset/aravind"  # Update 'aravind' to your subject name
BASE_OUTPUT = "output"

EPOCHS_LIST = [10, 20, 30, 40, 50]

def run_experiment(epochs):
    exp_name = f"exp_e{epochs}"
    output_dir = os.path.join(BASE_OUTPUT, exp_name)
    log_file = os.path.join(output_dir, "train.log")

    os.makedirs(output_dir, exist_ok=True)

    python_exe = sys.executable

    cmd = (
        f'"{python_exe}" diffusers/examples/text_to_image/train_text_to_image_lora.py '
        f'--pretrained_model_name_or_path={BASE_MODEL} '
        f'--train_data_dir="{DATASET_DIR}" '
        f'--resolution=512 '
        f'--train_batch_size=1 '
        f'--num_train_epochs={epochs} '
        f'--learning_rate=1e-4 '
        f'--rank=8 '
        f'--gradient_checkpointing '
        f'--mixed_precision=fp16 '
        f'--output_dir="{output_dir}" '
        f'--seed=42'
    )

    print(f"\nRunning experiment: {exp_name}")

    with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.PIPE,
            text=True
        )

        _, err = process.communicate()

        if process.returncode != 0:
            print("❌ Training failed with error:\n")
            print(err)
            raise RuntimeError("Training crashed")

if __name__ == "__main__":
    for e in EPOCHS_LIST:
        run_experiment(e)
