# Ouro Reproduction Notes

## Goal

Reproduce Ouro-style LoopLM training and SFT behavior with open components available in this workspace.

## Public references

- Paper: `arXiv:2510.25741` (Scaling Latent Reasoning via Looped Language Models)
- HF model card: `ByteDance/Ouro-1.4B` and `ByteDance/Ouro-1.4B-Thinking`

## Mapping: paper -> local files

- Looped decoder model:
  - `models/modeling_ouro.py`
- Config + presets:
  - `models/configuration_ouro.py`
  - `models/ouro_config.py`
- Multi-step CE + exit-gate + entropy-regularized objective:
  - `models/looplm_train.py`
- Pretraining launcher (single-node to multi-node shell):
  - `train/pretrain.py`
- Dataset streaming:
  - `data/streaming_lm.py`
- Stage configs:
  - `train/stages/stage1_stable.yaml`
  - `train/stages/stage2_ct_anneal.yaml`
  - `train/stages/stage3_longct.yaml`
  - `train/stages/stage4_midtrain.yaml`

## Known non-reproducible parts

- ByteDance internal `flame` / `torchtitan` code path.
- Exact 7.7T data shards and decontamination pipeline.
- Internal in-house reasoning evaluation harness.

## Practical approximation strategy

1. Validate architecture and objective on small-scale runs.
2. Run subset pretrain on open corpora with paper-shaped schedules.
3. SFT with public reasoning datasets to obtain thinking behavior.
4. Evaluate against public HF Ouro checkpoints.
