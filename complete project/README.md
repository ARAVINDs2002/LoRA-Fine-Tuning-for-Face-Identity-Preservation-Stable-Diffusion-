# LoRA Training & Evaluation

A comprehensive pipeline for training Stable Diffusion LoRA models with automated testing and face similarity evaluation using DeepFace.

## 📋 Overview

This project provides tools to:
- **Train** multiple LoRA models with different epoch configurations
- **Generate** test images from trained LoRA models
- **Evaluate** face similarity using industry-standard ArcFace embeddings

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 6GB+ VRAM)
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd lora_train
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Clone the diffusers repository** (required for training script)
```bash
git clone https://github.com/huggingface/diffusers
```

## 📁 Project Structure

```
lora_train/
├── trainv1.py              # Main training script
├── test.py                 # LoRA testing & image generation
├── similarity.py           # Face similarity evaluation
├── requirements.txt        # Python dependencies
├── lora_dataset/           # Your training images
│   └── aravind/           # Subject folder (customize name)
├── output/                 # Trained LoRA models as safetensors
│   ├── exp_e10/
│   ├── exp_e20/
│   └── ...
├── generated_images/       # Test outputs 
└── diffusers/             # Diffusers repo 
```

## 🎯 Usage

### 1. Prepare Your Dataset

Place 5-20 high-quality images of your subject in:
```
lora_dataset/aravind/
```

**Tips:**
- Use diverse poses and lighting
- 512x512 or higher resolution
- Clear, high-quality photos

### 2. Train LoRA Models

Update the `DATASET_DIR` in `trainv1.py` if you renamed the folder:
```python
DATASET_DIR = "lora_dataset/your_name"
```

Run training:
```bash
python trainv1.py
```

This will train 5 models with **10, 20, 30, 40, and 50 epochs** and save them to `output/exp_eXX/`.

### 3. Generate Test Images

Run the test script to generate images from each trained LoRA:
```bash
python test.py
```

Outputs 10 images per model to `generated_images/exp_eXX/`.

### 4. Evaluate Similarity

Measure face similarity between generated and real images:
```bash
python similarity.py
```

Results show **average, min, and max** similarity scores using ArcFace embeddings.

## ⚙️ Configuration

### Training Parameters (`trainv1.py`)

```python
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = "lora_dataset/aravind"
EPOCHS_LIST = [10, 20, 30, 40, 50]  # Customize epochs
```

### Testing Parameters (`test.py`)

```python
IMAGES_PER_LORA = 10        # Images to generate per model
WIDTH = 512
HEIGHT = 512
STEPS = 35                   # Inference steps
GUIDANCE = 7.0              # CFG scale
SEED_BASE = 1234            # For reproducibility
```

### Similarity Evaluation (`similarity.py`)

```python
MODEL_NAME = "ArcFace"      # Face recognition model
METRIC = "cosine"           # Distance metric
```

## 📊 Expected Results

After running all scripts, you'll have:
- ✅ 5 trained LoRA models in `output/`
- ✅ 50 generated test images (10 per model)
- ✅ Similarity scores showing model quality

**Sample Output:**
```
📊 exp_e10 → Avg: 0.652 | Min: 0.589 | Max: 0.712
📊 exp_e20 → Avg: 0.731 | Min: 0.688 | Max: 0.789
📊 exp_e30 → Avg: 0.768 | Min: 0.721 | Max: 0.823
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `train_batch_size` in `trainv1.py`
- Use `--gradient_checkpointing` (already enabled)
- Close other GPU applications

### Face Detection Fails
- Ensure images have clear, visible faces
- Check image resolution (min 256x256)
- Verify lighting quality

### Training Crashes
- Check `output/exp_eXX/train.log` for errors
- Ensure diffusers repo is cloned
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

## 📝 Notes

- First run will download ~4GB of models (Stable Diffusion + ArcFace)
- Training time: ~10-60 minutes per experiment (GPU-dependent)
- Generated images use the same prompt/seed for fair comparison

## 🙏 Credits

- [Diffusers](https://github.com/huggingface/diffusers) - Training framework
- [DeepFace](https://github.com/serengil/deepface) - Face similarity
- [Stable Diffusion](https://stability.ai/) - Base model

## 📄 License

This project is open-source and available under the MIT License.
