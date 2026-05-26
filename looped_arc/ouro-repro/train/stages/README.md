## Stage Configs

These YAML files provide paper-shaped schedules for subset or full runs:

- `stage1_stable.yaml`
- `stage2_ct_anneal.yaml`
- `stage3_longct.yaml`
- `stage4_midtrain.yaml`
- `stage1_scratch_custom_tokenizer.yaml` (template for brand-new tokenizer)

## Scratch vs continued pretrain

`train/pretrain.py` now follows tokenizer-first wiring (same idea as SmolLM nanotron configs):

- `tokenizer_name_or_path`: tokenizer source
- `vocab_size`: must match `len(tokenizer)` exactly
- `init_from_scratch: true`: initialize model weights from scratch

SmolLM uses the same explicit pairing in config:

- `text/pretraining/smollm1/config_smollm1_1B.yaml`:
  - `tokenizer.tokenizer_name_or_path: HuggingFaceTB/cosmo2-tokenizer`
  - `model.model_config.vocab_size: 49152`
- `text/pretraining/smollm3/stage1_8T.yaml`:
  - `tokenizer.tokenizer_name_or_path: meta-llama/Llama-3.2-1B`
  - `model.model_config.vocab_size: 128256`

If you need to train your own tokenizer first, use:

- `/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/data/tokenize/README.md`

### 8-GPU subset launch

```bash
bash /home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/train/run_subset_8gpu.sh stage1_stable
```

Repeat for each stage. The pretrain script emits loss and mean exit-step diagnostics for W&B logging integration.
