# SmolVLM2: Fine-Tuning Compact Vision-Language Models for Images & Videos

A lightweight framework for fine-tuning small, efficient vision-language models on both image and video datasets. Built for flexibility and performance, with support for parameter-efficient methods (LoRA/PEFT) and optional DeepSpeed integration.

## Features

- Efficient training of compact models (256M & 500M variants)
- Unified support for image and video modalities
- Parameter-Efficient Fine-Tuning (LoRA, PEFT)
- DeepSpeed Zero2/Zero3 support for large-scale training
- Easy freezing/unfreezing of vision & language submodules
- Prebuilt scripts for SLURM clusters and local interactive sessions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/smolvlm2.git
cd smollm2/vision/smolvlm

# Install dependencies
pip install -r requirements.txt
```

Or with Conda:

```bash
conda create -n smolvlm python=3.10
conda activate smolvlm
pip install -r requirements.txt
```

## Quick Start

### SLURM Multi-node Training

```bash
# Example: 16 nodes x 8 GPUs
bash scripts/train/multinode.sh
```

### Local Interactive Run (Single GPU)

```bash
bash scripts/train/interactive_256M.sh
```

### Custom Python Invocation

```bash
python smolvlm/train/train_mem.py \
  --model_name_or_path HuggingFaceTB/SmolVLM-256M-Instruct \
  --data_mixture scripts/mixtures/my_dataset.yaml \
  --output_dir checkpoints/my_run \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  ... other args ...
```

## Repository Layout

```
├── scripts/             # Training scripts (SLURM & local)
├── smolvlm/             # Model code and training logic
├── scripts/mixtures/    # Data mixture configurations
├── checkpoints/         # Default output directory
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your enhancements or bug fixes.

## License

Apache 2.0
