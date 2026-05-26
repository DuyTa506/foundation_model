import os
import math
import random
import logging
import yaml
from typing import Dict, Any, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, ConcatDataset

from smolvlm.datasets.dataset import SupervisedDataset
from smolvlm.train.args import DataArguments, TrainingArguments, ModelArguments
from smolvlm.constants import IGNORE_INDEX
from smolvlm.utils import mprint  # or your custom printing utility
from tabulate import tabulate
from transformers import ProcessorMixin
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


##############################################################################
# Multi-subsequence "varlen" packing with integer-coded `subseq_ids`
##############################################################################

def len2weight(num_effective_tokens: int, mode: str) -> float:
    """
    Returns the sub-sample weight given the sub-sequence length.
    """
    if num_effective_tokens == 0:
        return 0.0  # or skip
    if mode == "token":
        return 1.0  # no length-based weighting
    elif mode == "sample":
        # each sub-sample counts equally, so 1 / length
        return 1.0 / num_effective_tokens
    elif mode == "square":
        # default in InternVL
        return 1.0 / (num_effective_tokens**0.5)
    else:
        # 'none' or fallback
        return 1.0


class PackedConcatDataset(ConcatDataset):
    """
    Merges multiple short sub-samples from an underlying ConcatDataset into a
    single “packed” sample, up to a `cutoff_len` tokens. Assigns integer-coded
    sub-sequence IDs in `subseq_ids`: 
        1 => sub-sample #1
        2 => sub-sample #2
        ...
    so your collator can turn them into block diagonal (varlen) attention masks.

    Each returned item from __getitem__ is:
        {
          "input_ids":  (sum_of_sub_len,) int,
          "labels":     (sum_of_sub_len,) int,
          "subseq_ids": (sum_of_sub_len,) int in [1..N],
          "pixel_values": (sum_of_frames, 3, H, W)  if images exist
        }
    We do NOT do final token-level padding to cutoff_len; we let the data collator
    handle batch-level padding (less wasteful).

    Attributes:
        datasets: List of sub-datasets we are merging.
        cutoff_len: Max tokens we want in a single “packed” sample. 
        pad_token_id: If needed for partial fix-ups.
        packed_cursor: Tracks how many sub-samples we have consumed so far.
    """

    def __init__(
        self,
        datasets: List,
        data_args,
        model_max_length: int = 2048
    ):
        super().__init__(datasets)
        self.data_args = data_args
        self.cutoff_len = max(model_max_length, 1)
        self.pad_token_id = getattr(data_args, "pad_token_id", 0)
        self.packed_cursor = 0

        logger.info(
            f"[PackedConcatDataset] Using cutoff_len={self.cutoff_len}; "
            f"we'll merge multiple sub-samples per item."
        )

    def __len__(self):
        # Underlying total number of sub-samples
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        We ignore 'idx' because each call to __getitem__ gets the “next” packed sample,
        from self.packed_cursor onward.
        """
        if self.packed_cursor >= len(self):
            raise IndexError("No more sub-samples left to pack in PackedConcatDataset.")

        # Accumulate sub-samples
        chunk_input_ids = []
        chunk_labels = []
        chunk_subseq_ids = []
        pixel_key = None
        pixel_values_list = []

        sub_seq_counter = 0
        current_token_count = 0

        while True:
            if self.packed_cursor >= len(self):
                break

            sub_item = super().__getitem__(self.packed_cursor)
            self.packed_cursor += 1

            sub_len = sub_item["input_ids"].size(0)
            if (current_token_count > 0) and (current_token_count + sub_len) > self.cutoff_len:
                # Revert if we can't fit this sub-sample
                self.packed_cursor -= 1
                break

            sub_seq_counter += 1
            seq_id_tensor = torch.full(
                (sub_len,),
                fill_value=sub_seq_counter,
                dtype=torch.long,
                device=sub_item["input_ids"].device
            )

            chunk_input_ids.append(sub_item["input_ids"])
            chunk_labels.append(sub_item["labels"])
            chunk_subseq_ids.append(seq_id_tensor)

            # If images are present
            if "pixel_values" in sub_item:
                pixel_key = "pixel_values"
                pixel_values_list.append(sub_item["pixel_values"])

            current_token_count += sub_len
            print("[Sequence Packing] current num tokens:", current_token_count)
            if current_token_count >= self.cutoff_len:
                break

        # Merge text
        if len(chunk_input_ids) == 0:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
            }

        merged_input_ids = torch.cat(chunk_input_ids, dim=0)
        merged_labels = torch.cat(chunk_labels, dim=0)
        merged_subseq_ids = torch.cat(chunk_subseq_ids, dim=0)

        # Merge images along frame dimension if present
        merged_pixel_values = None
        if pixel_key and pixel_values_list:
            merged_pixel_values = torch.cat(pixel_values_list, dim=0)
            # shape => (f1+f2+..., 3, H, W)

        loss_weight = torch.ones_like(merged_subseq_ids, dtype=torch.float32)
        unique_ids = merged_subseq_ids.unique()
        unique_ids = unique_ids[unique_ids > 0]  # ignore pad=0
        for sid in unique_ids.tolist():
            mask = (merged_subseq_ids == sid)
            num_eff = (merged_labels[mask] != IGNORE_INDEX).sum().item()
            w = len2weight(num_eff, self.data_args.loss_reduction)
            loss_weight[mask] = w
        
        # Build final
        out_dict = {
            "input_ids": merged_input_ids,
            "labels": merged_labels,
            "attention_mask": merged_subseq_ids,
            "loss_weight": loss_weight,  
        }
        if merged_pixel_values is not None:
            out_dict[pixel_key] = merged_pixel_values

        return out_dict


# Varlen Collator (subseq_ids => block diagonal)
##############################################################################


def pad_sequence_varlen(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Similar signature to torch.nn.utils.rnn.pad_sequence, but treats each Tensor
    as a 1D sequence of any integer-coded or float tokens, and uses integer-coded
    varlen semantics if desired.

    If batch_first=True, returns (batch_size, seq_len).
    Otherwise returns (seq_len, batch_size).
    """

    if len(sequences) == 0:
        # Return an empty tensor if no sequences
        return torch.tensor([], dtype=torch.long)

    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    device = sequences[0].device
    dtype = sequences[0].dtype

    if batch_first:
        # Shape => (batch_size, max_len)
        out = torch.full((batch_size, max_len), padding_value, device=device, dtype=dtype)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            out[i, :length] = seq
    else:
        # Shape => (max_len, batch_size)
        out = torch.full((max_len, batch_size), padding_value, device=device, dtype=dtype)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            out[:length, i] = seq

    return out

##############################################################################
# Standard Collator (0/1 attn mask)
##############################################################################

class DataCollatorForSupervisedDataset:
    """
    Collates examples containing text-only or text+image/video data.

    1) Text sequences (input_ids, attention_mask, labels) are padded to the maximum batch length.
       - If model_max_length is set, we optionally truncate.
    2) Pixel data (pixel_values, optional) is padded to (max_frames, 3, max_h, max_w).
    3) Pixel-level mask (pixel_attention_mask):
       - If provided in the example, we pad it accordingly.
       - If not provided, we fill the valid image region with 1, the remainder with 0.
    """

    def __init__(
        self,
        pad_token_id: int,
        model_max_length: Optional[int] = None,
        image_size: Optional[int] = None,
        func_pad_sequence = pad_sequence,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = IGNORE_INDEX
        self.model_max_length = model_max_length
        self.image_size = image_size
        self.func_pad_sequence = func_pad_sequence

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ################################################################
        # PART A: Pad the text data (input_ids, attention_mask, labels)
        ################################################################
        
        attention_masks_list = []
        for ex in examples:
            # If "attention_mask" is missing, we generate it on the fly
            if "attention_mask" in ex:
                attention_masks_list.append(ex["attention_mask"])
            else:
                am = (ex["input_ids"] != self.pad_token_id).long()
                attention_masks_list.append(am)


        input_ids = self.func_pad_sequence(
            [ex["input_ids"] for ex in examples],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        attention_mask = self.func_pad_sequence(
            attention_masks_list,
            batch_first=True,
            padding_value=0
        )
        labels = self.func_pad_sequence(
            [ex["labels"] for ex in examples],
            batch_first=True,
            padding_value=self.ignore_index
        )

        # Optional: truncate if model_max_length is specified
        if self.model_max_length and input_ids.size(1) > self.model_max_length:
            input_ids = input_ids[:, :self.model_max_length]
            attention_mask = attention_mask[:, :self.model_max_length]
            labels = labels[:, :self.model_max_length]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        ################################################################
        # PART B: Handle pixel data (pixel_values) + pixel_attention_mask
        ################################################################
        # Step 1: figure out maximum frames, height, width across the batch
        pvs = [ex["pixel_values"] for ex in examples if "pixel_values" in ex]
        if pvs:  # there is at least one non-None pixel_values
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            max_h = max_w = self.image_size
            max_frames = 1 #TODO: verify this is good default

        # Step 2: create padded pixel_values and pixel_attention_mask for each example
        padded_pixel_values_list = []
        padded_pixel_mask_list = []

        for ex in examples:
            pv = ex.get("pixel_values", None)
            pm = ex.get("pixel_attention_mask", None)  # shape (f, h, w) if provided

            if pv is None:
                # text-only => fill pixel data + mask with zeros
                shape_pv = (max_frames, 3, max_h, max_w)
                shape_pm = (max_frames, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
                padded_pm = torch.zeros(shape_pm, dtype=torch.long)
            else:
                f, c, h, w = pv.shape
                # Prepare final storage
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device
                )
                padded_pm = torch.zeros(
                    (max_frames, max_h, max_w),
                    dtype=torch.long,
                    device=pv.device
                )

                padded_pv[:f, :, :h, :w] = pv
                # Copy or fill the pixel attention mask
                if pm is not None:
                    padded_pm[:f, :h, :w] = pm
                else:
                    # Mark valid region as 1
                    padded_pm[:f, :h, :w] = 1

            padded_pixel_values_list.append(padded_pv)
            padded_pixel_mask_list.append(padded_pm)

        # Finally, stack along batch dimension
        ## try not outputting pixel_values in text-only sample
        #if any("pixel_values" in ex for ex in examples):
        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

##############################################################################
# Summaries
##############################################################################
def display_overview(summary_data: Dict[str, Any], total_count: int) -> None:
    print("=== Overview ===")
    print(f"Aggregate Sample Count: {total_count}\n")
    for category, info in summary_data.items():
        ctotal = info["total_samples"]
        cpct = (ctotal / total_count * 100) if total_count > 0 else 0
        print(f"{category.title()} Overview")
        print(f"Number of Samples: {ctotal} ({cpct:.2f}%)")
        print("-" * 50)
        table_data = []
        headers = ["Dataset", "Count", "Percentage"]
        for entry in info["datasets"]:
            esamples = entry["samples"]
            epct = (esamples / total_count * 100) if total_count > 0 else 0
            table_data.append([entry["dataset_name"], esamples, f"{epct:.2f}%"])
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print()


##############################################################################
# Main builder logic
##############################################################################
def build_datasets(
    data_args: DataArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    processor: ProcessorMixin,
    split: str = "train",
):
    """
    1) Load a YAML describing multiple sub-datasets.
    2) Create a list of SupervisedDataset objects.
    3) If data_args.packed => use PackedConcatDataset (with subseq_ids),
       else => normal ConcatDataset.
    """
    if getattr(model_args, "frames_per_clip", 1) > 1:
        from smolvlm.datasets.dataset_clip_sampling import SupervisedDataset
    else:
        from smolvlm.datasets.dataset import SupervisedDataset
    
    mprint(f"[Dataset-INFO]: Loading from {data_args.data_mixture}")
    with open(data_args.data_mixture, "r") as yf:
        meta_datasets = yaml.safe_load(yf)

    all_datasets = []
    extra_info = []

    for dataset_type, dataset_list in meta_datasets.items():
        for ds_args in dataset_list:
            ds = SupervisedDataset(
                dataset_args=ds_args,
                processor=processor,
                data_args=data_args,
                training_args=training_args,
                model_args=model_args,
            )
            all_datasets.append(ds)
            extra_info.append({
                "dataset_name": ds.name,
                "modality": ds.modality,
                "samples": len(ds),
            })

    # Summaries
    from collections import defaultdict
    modality_summary = defaultdict(lambda: {"total_samples": 0, "datasets": []})
    total_samples = 0
    for entry in extra_info:
        mod = entry["modality"]
        dsname = entry["dataset_name"]
        samples = entry["samples"]
        modality_summary[mod]["total_samples"] += samples
        modality_summary[mod]["datasets"].append(entry)
        total_samples += samples

    display_overview(modality_summary, total_samples)

    # Build final dataset
    if data_args.packed:
        mprint("[build_datasets] Using PackedConcatDataset for multi-sample packing with subseq_ids.")
        dataset = PackedConcatDataset(
            all_datasets,
            data_args=data_args,
            model_max_length=training_args.model_max_length
        )
    else:
        mprint("[build_datasets] Using standard ConcatDataset (no packing).")
        dataset = ConcatDataset(all_datasets)

    # Save some info in training_args
    training_args.sample_lens = [e["samples"] for e in extra_info]
    training_args.data_info = extra_info

    return dataset


def make_supervised_data_module(
    processor: ProcessorMixin,
    data_args: DataArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
):
    """
    Creates train_dataset, eval_dataset, and data_collator.
    If data_args.packed => we do integer-coded subseq_ids approach,
    else => normal approach with 0/1 attention_mask.
    """
    train_dataset = build_datasets(data_args, training_args, model_args, processor, split="train")
    eval_dataset = None

    if data_args.packed:
        # Use the varlen collator
        data_collator         = DataCollatorForSupervisedDataset(
            pad_token_id      = processor.tokenizer.pad_token_id,
            model_max_length  = processor.tokenizer.model_max_length,
            image_size        = data_args.video_target_size,
            func_pad_sequence = pad_sequence_varlen,
        )
    else:
        # Use the normal collator
        data_collator         = DataCollatorForSupervisedDataset(
            pad_token_id      = processor.tokenizer.pad_token_id,
            model_max_length  = processor.tokenizer.model_max_length,
            image_size        = data_args.video_target_size,
            func_pad_sequence = pad_sequence,
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )