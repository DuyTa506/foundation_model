## Ouro Reproduction Workspace

This folder implements a practical reproduction stack for `Ouro-1.4B-Thinking`:

- Model-side parity work (looped full-stack decoder, exit gates, multi-step loss)
- Training shell adapted from `hierachical_arc/HRM-Text` (FSDP2-oriented flow)
- Causal streaming dataloader adapted from `looped_arc/OpenMythos`
- Stage configs and SFT/eval helpers

### Layout

- `models/`: model definitions and loop-specific training loss
- `train/`: pretrain and SFT launch scripts/configs
- `data/`: streaming causal text dataloader + tokenization helpers
- `eval/`: parity and benchmark runners
- `docs/`: reproduction notes and paper-to-config mapping

### Scope

This project aims to be:

1. Architecturally faithful to the public Ouro paper/model card where possible.
2. Operational on a single node for smoke tests and subset pretraining.
3. Explicit about gaps that require unreleased internal code or multi-node scale.
