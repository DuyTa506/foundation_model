from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .modeling_ouro import OuroForCausalLM


IGNORE_INDEX = -100


@dataclass
class LoopLMLossConfig:
    kl_beta: float = 0.1
    include_adaptive_exit_loss: bool = False
    adaptive_exit_weight: float = 0.1
    step_loss_weights: Optional[list[float]] = None
    early_exit_threshold: float = 1.0


def _build_hazard(exit_logits: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.sigmoid(logits) for logits in exit_logits]


def _build_exit_distribution(hazards: list[torch.Tensor]) -> list[torch.Tensor]:
    # p_t = hazard_t * survival_{t-1}
    probs = []
    survival = torch.ones_like(hazards[0])
    for hz in hazards:
        p_t = hz * survival
        probs.append(p_t)
        survival = survival * (1.0 - hz)
    residual = survival
    # Keep last-step mass normalized by adding residual to last step.
    probs[-1] = probs[-1] + residual
    return probs


def _uniform_prior(num_steps: int, like: torch.Tensor) -> torch.Tensor:
    return torch.full((num_steps,) + like.shape, 1.0 / num_steps, device=like.device, dtype=like.dtype)


def _stack_probs(step_probs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(step_probs, dim=0).clamp_min(1e-8)


def _token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits.float(), flat_labels.long(), ignore_index=IGNORE_INDEX, reduction="none")
    return ce.reshape_as(labels)


class OuroLoopLMTrain(nn.Module):
    def __init__(self, model: OuroForCausalLM, config: LoopLMLossConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.exit_head = nn.Linear(model.config.hidden_size, 1, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        total_ut_steps: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        step_hidden = self.model.model(input_ids, total_ut_steps=total_ut_steps, return_all_steps=True)
        step_logits = [self.model.lm_head(h) for h in step_hidden]

        num_steps = len(step_logits)
        step_loss_weights = self.config.step_loss_weights
        if step_loss_weights is None:
            step_loss_weights = [1.0 / num_steps] * num_steps

        valid_mask = (labels != IGNORE_INDEX).float()
        denom = valid_mask.sum().clamp_min(1.0)

        # Step CE
        step_ce = []
        for i, logits in enumerate(step_logits):
            token_ce = _token_ce(logits, labels) * valid_mask
            loss_i = token_ce.sum() / denom
            step_ce.append(loss_i * step_loss_weights[i])
        ce_loss = torch.stack(step_ce).sum()

        # Exit distribution from hidden states
        exit_logits = [self.exit_head(h).squeeze(-1) for h in step_hidden]
        hazards = _build_hazard(exit_logits)
        exit_probs = _build_exit_distribution(hazards)
        stacked_probs = _stack_probs(exit_probs)  # [T, B, S]

        # KL(stacked_probs || uniform)
        uniform = _uniform_prior(num_steps, stacked_probs[0])
        # Only compute over valid labels.
        valid = valid_mask.unsqueeze(0)
        kl = (stacked_probs * (stacked_probs.log() - uniform.log())) * valid
        kl_loss = kl.sum() / valid.sum().clamp_min(1.0)

        loss = ce_loss + self.config.kl_beta * kl_loss

        adaptive_exit_loss = torch.tensor(0.0, device=loss.device)
        if self.config.include_adaptive_exit_loss:
            with torch.no_grad():
                # Approximate target: earlier exit when later-step CE improvements are small.
                per_step_ce = torch.stack([_token_ce(l, labels) for l in step_logits], dim=0)  # [T, B, S]
                step_delta = per_step_ce[:-1] - per_step_ce[1:]
                improve = torch.cat([step_delta, torch.zeros_like(step_delta[:1])], dim=0)
                # Higher improve -> continue; lower improve -> exit now
                target_hazard = torch.sigmoid(-improve)
                target_hazard = target_hazard * valid

            hazard_stack = torch.stack(hazards, dim=0)
            bce = F.binary_cross_entropy(hazard_stack, target_hazard, reduction="none")
            adaptive_exit_loss = bce.sum() / valid.sum().clamp_min(1.0)
            loss = loss + self.config.adaptive_exit_weight * adaptive_exit_loss

        # Diagnostics
        mean_exit_step = (
            torch.stack(
                [(i + 1) * exit_probs[i] for i in range(num_steps)],
                dim=0,
            ).sum(dim=0)
            * valid_mask
        ).sum() / denom

        return {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "adaptive_exit_loss": adaptive_exit_loss.detach(),
            "mean_exit_step": mean_exit_step.detach(),
        }

