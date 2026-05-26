from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset


class StreamingCausalLMDataset(IterableDataset):
    """
    Streaming text dataset for causal LM.

    Emits:
    - input_ids: [seq_len]
    - labels: [seq_len] (shifted by 1; last token ignored)
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        dataset_config: str | None,
        split: str,
        text_field: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def _iter_stream(self) -> Iterable[dict]:
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
        )
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1

        total_shards = max(1, self.world_size * num_workers)
        shard_index = self.rank * num_workers + worker_id
        yield from ds.shard(num_shards=total_shards, index=shard_index)

    def __iter__(self):
        token_buffer: list[int] = []
        for row in self._iter_stream():
            text = row.get(self.text_field)
            if not text:
                continue
            token_buffer.extend(self.tokenizer.encode(text))

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                del token_buffer[: self.seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}

