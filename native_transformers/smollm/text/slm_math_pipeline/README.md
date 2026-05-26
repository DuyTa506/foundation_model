# SLM Math Pipeline (EN + VI)

This module provides a baseline 3-stage pipeline:

1. **Pretrain**: Megatron-LM + DeepSpeed on EN+VI math/science corpus.
2. **Finetune**: TRL SFT + LoRA on instruction-style math/science data.
3. **Posttrain**: TRL DPO (or fallback posttrain-SFT) for alignment.

The baseline is designed for **8xH200** training setups.

## Directory layout

- `docs/framework_survey.md`: framework comparison and selected stack rationale.
- `configs/`: model, dataset, and stage-specific training configs.
- `scripts/`: reproducible scripts for dataset curation, launch, and eval.

## Environment setup (reproducible)

```bash
cd native_transformers/smollm/text/slm_math_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Megatron-LM (required for pretraining launcher):

```bash
cd /path/to/your/workspace
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

Optional but recommended for exact reproducibility:
- capture the fully resolved environment after install:
  - `pip freeze > requirements.lock.txt`
- store CUDA/NVIDIA versions with your run logs.

## Data sources (initial baseline)

### Pretrain mix (web + specialist math)
- `HuggingFaceTB/finemath`
- `openbmb/UltraData-Math`
- `open-web-math/open-web-math`
- `epfml/FineWeb2-HQ` (including `vie_Latn` slices)
- `uonlp/CulturaX`
- `VTSNLP/vietnamese_curated_dataset`
- `Symato/madlad-400_vi`
- `Symato/c4_vi-filtered_200GB`
- `Symato/hplt-vi`

### Finetune / posttrain mix
- `agicorp/MathInstruct`
- `meta-math/MetaMathQA`
- `hkust-nlp/dart-math-pool-math`
- Vietnamese math instruction collection candidates:
  - `5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated`
  - `5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated`
  - `5CD-AI/Vietnamese-nvidia-OpenMathInstruct-1-50k-gg-translated`
  - `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated`
- optional science supervision/eval data:
  - `allenai/sciq`
  - `derek-thomas/ScienceQA`
  - `allenai/openbookqa`
  - `allenai/ai2_arc`

## Quick start

### 1) Build deterministic dataset manifest
```bash
python scripts/build_dataset_index.py \
  --config configs/datasets_en_vi_math_pretrain.yaml \
  --output_dir outputs/pretrain_manifest
```

### 2) Run language filtering and near-dedup
```bash
python scripts/filter_language_en_vi.py \
  --input_manifest outputs/pretrain_manifest/dataset_manifest.jsonl \
  --output_manifest outputs/pretrain_manifest/dataset_manifest_lang.jsonl

python scripts/dedup_min_hash.py \
  --input_manifest outputs/pretrain_manifest/dataset_manifest_lang.jsonl \
  --output_manifest outputs/pretrain_manifest/dataset_manifest_lang_dedup.jsonl
```

### 3) Prepare Megatron indexed dataset
```bash
python scripts/prepare_megatron_indexed_dataset.py \
  --manifest outputs/pretrain_manifest/dataset_manifest_lang_dedup.jsonl \
  --base_model openbmb/MiniCPM5-1B \
  --output_prefix outputs/pretrain_data/en_vi_math
```

### 4) Launch pretraining (8xH200)
```bash
bash scripts/launch_pretrain_megatron_ds.sh \
  --config configs/training_8xH200_megatron_ds_pretrain.yaml
```

### 5) Launch finetune
```bash
python scripts/launch_finetune_trl_sft.py \
  --training_config configs/training_finetune_trl_sft.yaml \
  --dataset_config configs/datasets_en_vi_math_finetune.yaml
```

### 6) Launch posttrain
```bash
python scripts/launch_posttrain_trl_dpo.py \
  --training_config configs/training_posttrain_trl_dpo.yaml \
  --dataset_config configs/datasets_en_vi_math_posttrain.yaml
```

### 7) Evaluate
```bash
python scripts/run_eval_lighteval.py \
  --model_path outputs/posttrain \
  --tasks gsm8k,math,arc_challenge,sciq
```

## Notes
- Keep quality first: language filtering, dedup, and decontamination happen before training.
- Start with conservative hyperparameters and scale global token batch gradually.
- Posttrain script automatically falls back to SFT when DPO pairs are unavailable.

## Reasoning tag policy (`<think>...</think>`)
- **Pretrain**: avoid explicit chain-of-thought tags; keep raw high-quality math/science text.
- **Finetune (default baseline)**: train with final answers/solutions, strip `<think>` traces if present.
- **Posttrain**:
  - For private-reasoning models: preference-train with hidden reasoning traces but do not expose traces in final response format.
  - For transparent-reasoning experiments: keep traces in a separate run and evaluate separately.
- Practical default in this module: scripts support stripping `<think>` tags before building train text.
