## Build a brand-new tokenizer (from pretrain corpus)

Yes, you need this before true scratch pretraining.

### 1) Train tokenizer from streaming corpus

```bash
python3 /home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/data/tokenize/train_bpe_tokenizer.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --text-field text \
  --max-rows 5000000 \
  --vocab-size 50000 \
  --out-dir /home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/artifacts/tokenizer_50k
```

### 2) Read actual vocab size

```bash
python3 /home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/data/tokenize/print_vocab_size.py \
  --tokenizer /home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/artifacts/tokenizer_50k
```

### 3) Plug into scratch pretrain config

Edit:

- `tokenizer_name_or_path`: your tokenizer folder
- `vocab_size`: output from step (2)
- `init_from_scratch: true`

Template config:

- `/home/duy/Downloads/duy_dev/foundation_model/looped_arc/ouro-repro/train/stages/stage1_scratch_custom_tokenizer.yaml`

### Notes

- `actual_vocab_size` may be slightly different from requested if corpus/sample constraints apply.
- Keep model `vocab_size` exactly equal to `len(tokenizer)` (the trainer enforces this).
