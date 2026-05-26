from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .configuration_ouro import OuroConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: OuroConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rope(position_ids)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, torch.finfo(attn_scores.dtype).min)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: OuroConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: OuroConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), position_ids)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class OuroModel(nn.Module):
    def __init__(self, config: OuroConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def run_one_stack(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        return self.final_norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        total_ut_steps: Optional[int] = None,
        return_all_steps: bool = True,
    ) -> list[torch.Tensor] | torch.Tensor:
        steps = total_ut_steps if total_ut_steps is not None else self.config.total_ut_steps
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        input_emb = self.embed_tokens(input_ids)

        hidden_states = input_emb
        step_outputs: list[torch.Tensor] = []
        for _ in range(steps):
            hidden_states = self.run_one_stack(hidden_states, position_ids)
            step_outputs.append(hidden_states)

        return step_outputs if return_all_steps else hidden_states


class OuroForCausalLM(nn.Module):
    def __init__(self, config: OuroConfig):
        super().__init__()
        self.config = config
        self.model = OuroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "OuroForCausalLM":
        config = OuroConfig(**config_dict)
        return cls(config)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        total_ut_steps: Optional[int] = None,
        return_all_steps: bool = True,
    ) -> list[torch.Tensor] | torch.Tensor:
        step_hidden = self.model(input_ids, total_ut_steps=total_ut_steps, return_all_steps=True)
        step_logits = [self.lm_head(h) for h in step_hidden]
        if return_all_steps:
            return step_logits
        return step_logits[-1]

