# Full 7.7T Cluster Runbook

This runbook operationalizes the Phase-3 to-do for full-scale reproduction.

## Target

- Model: Ouro-style 1.4B LoopLM (`total_ut_steps=4`)
- Token budget: 7.7T across 4 stages
- Parallelism: multi-node, 64–128 GPUs recommended

## Stages

1. Stage-1 stable pretrain (4k context)
2. Stage-2 CT anneal (16k context)
3. Stage-3 LongCT (64k context)
4. Stage-4 Mid-train (32k context)

## Infra requirements

- Distributed launcher: `torchrun` multi-node
- Storage: 15–30 TB for tokenized shards and checkpoints
- Checkpoint cadence: every 1k–5k optimizer steps
- Logging: W&B project group per stage

## Upcycling branch to 2.6B

After stable 1.4B convergence, branch by depth-upcycling and continue stage schedule.

## Acceptance criteria

- Non-collapsed exit histogram
- Stable loss trajectory through stage transitions
- Reasoning uplift from `total_ut_steps=1` to `4` on held-out eval slices
- Final SFT checkpoint with parity envelope near paper order-of-magnitude

