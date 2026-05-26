# varlen_packing.py
# ---------------------------------------------------
# A universal patch for "flash_attention_2" to handle
# multi-subsequence "packed" sequences via integer-coded
# or block-coded 1D attention maskszq
# ---------------------------------------------------

import torch
import torch.nn.functional as F
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _get_seqlens_in_batch(mask_1d: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1D integer-coded mask (like [1,1,1,2,2,2,2,3,3,3,0,0,...])
    into sub-sequence lengths. We assume sub-sequence IDs appear in ascending
    order but do not revisit older IDs. Each contiguous run of a nonzero ID
    counts toward that sub-sequence's length.

    Example:
      mask_1d = [1,1,1,2,2,2,0,0]
      => sub-seq #1 => length=3, #2 => length=3
      => lengths = [3, 3]

    Returns a 1D int32 of sub-sequence lengths, e.g. [3,3].
    """
    mask_1d = mask_1d.view(-1)  # flatten

    # Filter out zeros (which is "padding")
    nonzero_mask = mask_1d[mask_1d != 0]
    if nonzero_mask.numel() == 0:
        # no real tokens
        return torch.tensor([], dtype=torch.int32)

    lengths = []
    count = 1
    last_id = nonzero_mask[0].item()

    for val in nonzero_mask[1:]:
        vid = val.item()
        if vid == last_id:
            count += 1
        else:
            lengths.append(count)
            last_id = vid
            count = 1

    if count > 0:
        lengths.append(count)

    return torch.tensor(lengths, dtype=torch.int32)


def get_unpad_data(attention_mask: torch.Tensor):
    """
    Our custom override for varlen "flash_attention_2". 
    Typically `_get_unpad_data` returns:
       (indices, cu_seqlens, max_seqlen_in_batch)

    We interpret `attention_mask` as a 2D or 1D integer-coded array:
      shape => (batch_size, seq_len) or (seq_len,)
    For each row, we parse sub-seq lengths => build cu_seqlens => build indices.

    Example for a single row [1,1,1,2,2,2,2,0,0]:
      => sub-seq #1 => length=3, sub-seq #2 => length=4
      => cu_seqlens => [0,3,7], max_len => 4
      => indices => positions that are !=0 => [0,1,2,3,4,5,6]
    If multiple rows => do row by row, then unify.

    We also forcibly move the returned Tensors to the same device as
    `attention_mask` if needed. (Essential for "cu_seqlens_q must be on CUDA".)
    """
    dev = attention_mask.device  # We'll force everything onto this device at the end
    #import ipdb; ipdb.set_trace()
    if attention_mask.dim() == 1:
        # Single row
        mask_flat = attention_mask
        lengths = _get_seqlens_in_batch(mask_flat)
        if lengths.numel() == 0:
            # no real tokens
            indices = torch.tensor([], dtype=torch.long, device=dev)
            cu_seqlens = torch.tensor([0], dtype=torch.int32, device=dev)
            return (indices, cu_seqlens, 0)

        cu_seqlens = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=dev),
            torch.cumsum(lengths, dim=0).to(dev)
        ], dim=0)

        max_len = lengths.max().item()
        indices = (mask_flat != 0).nonzero().squeeze(-1).to(dev)

        return (indices, cu_seqlens, max_len)

    elif attention_mask.dim() == 2:
        bsz, seqlen = attention_mask.shape

        indices_list = []
        cu_seqlens_list = [0]
        current_offset = 0
        max_len = 0

        for row_idx in range(bsz):
            row = attention_mask[row_idx]
            lengths = _get_seqlens_in_batch(row)
            if lengths.numel() > 0:
                new_cu = torch.cumsum(lengths, dim=0) + cu_seqlens_list[-1]
                cu_seqlens_list.extend(new_cu.tolist())
                row_max = lengths.max().item()
                if row_max > max_len:
                    max_len = row_max
            else:
                # no real tokens => skip
                pass

            row_indices = (row != 0).nonzero().squeeze(-1) + current_offset
            indices_list.append(row_indices)
            current_offset += seqlen

        if len(cu_seqlens_list) == 1:
            # means no real tokens at all
            indices = torch.tensor([], dtype=torch.long, device=dev)
            cu_seqlens = torch.tensor([0], dtype=torch.int32, device=dev)
            return (indices, cu_seqlens, 0)

        # Build final Tensors
        indices = torch.cat(indices_list, dim=0).to(dev)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dev)

        return (indices, cu_seqlens, max_len)

    else:
        raise ValueError(
            f"_my_get_unpad_data_varlen expects dim=1 or 2, got shape {attention_mask.shape}"
        )


def apply_varlen_patch():
    """
    Monkey-patch HF's `_get_unpad_data` with `_my_get_unpad_data_varlen`.
    This modifies the varlen logic for "flash_attention_2".
    """
    try:
        from transformers import modeling_flash_attention_utils
    except ImportError:
        logger.warning(
            "apply_varlen_patch: transformers>=4.45 needed for flash_attention_2. Not patching."
        )
        return None

    if not hasattr(modeling_flash_attention_utils, "_get_unpad_data"):
        logger.warning(
            "apply_varlen_patch: can't find `_get_unpad_data` in modeling_flash_attention_utils. "
            "Your Transformers version might not have flash_attn varlen logic."
        )
        return None

    # Replace
    old_func = modeling_flash_attention_utils._get_unpad_data
    modeling_flash_attention_utils._get_unpad_data = get_unpad_data

    logger.info(
        "apply_varlen_patch: Replaced `_get_unpad_data` with our varlen integer-coded approach."
    )
    return old_func  # If you want to restore it later