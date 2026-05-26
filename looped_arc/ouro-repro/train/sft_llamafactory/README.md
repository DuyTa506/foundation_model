## LlamaFactory SFT (Thinking)

This directory provides a smoke-test SFT recipe to reproduce a thinking-style variant.

### Supported bases

- `ByteDance/Ouro-1.4B` (public base)
- local base checkpoint converted to HF format

### Expected chat formatting

Use the same chat template style as Ouro:

- system message + chat turns
- generation prompt includes assistant prefix
- optional `enable_thinking` behavior

### Launch (example)

```bash
llamafactory-cli train train/sft_llamafactory/sft_ouro_1_4b.yaml
```

### Note

This stage is intended as a practical open approximation. Exact internal decontamination and private blends from the paper are not available publicly.
