"""
Math answer verification and reward functions for GRPO/RLVR.

Provides:
  - extract_answer(text) -> str | None
  - answers_equal(pred, gold) -> bool  (sympy-based math equivalence)
  - correctness_reward(completions, ground_truths) -> list[float]
  - format_reward(completions) -> list[float]
  - language_consistency_reward(completions, prompts) -> list[float]
"""

from __future__ import annotations

import re
from typing import Sequence


# ─── Answer extraction ────────────────────────────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
_THE_ANSWER_RE = re.compile(
    r"(?:the answer is|answer:|=|:)\s*([+-]?\d[\d\s,./]*\d?|\d)",
    re.IGNORECASE,
)
_LAST_NUMBER_RE = re.compile(r"([+-]?\d[\d\s,./]*\d|\d)")


def extract_answer(text: str) -> str | None:
    """
    Extract the final answer from a model completion.
    Priority: \\boxed{} > "the answer is X" > last number in the text.
    """
    if not text:
        return None

    # 1. LaTeX boxed
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()

    # 2. "the answer is ..." or "Answer: ..."
    m = _THE_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "").replace(" ", "")

    # 3. Last number-like token in text
    numbers = _LAST_NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip().replace(",", "").replace(" ", "")

    return None


def _normalize(s: str) -> str:
    """Strip whitespace, commas, trailing zeros."""
    return s.strip().replace(",", "").replace(" ", "").rstrip("0").rstrip(".")


def answers_equal(pred: str | None, gold: str | None) -> bool:
    """
    Math-aware equality: tries string normalization first, then sympy.
    1/2 == 0.5 == 0.50 -> True
    """
    if pred is None or gold is None:
        return False

    pred_n = _normalize(pred)
    gold_n = _normalize(gold)

    if pred_n == gold_n:
        return True

    # Try float comparison
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        pass

    # Sympy symbolic equality (handles 1/2 == 0.5, sqrt(2)*sqrt(2) == 2, etc.)
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex

        def _parse(s: str):
            try:
                return parse_latex(s)
            except Exception:
                return sympify(s)

        diff = simplify(_parse(pred_n) - _parse(gold_n))
        return diff == 0
    except Exception:
        pass

    return False


# ─── Reward functions ─────────────────────────────────────────────────────────

def correctness_reward(
    completions: Sequence[str],
    ground_truths: Sequence[str],
    reward_correct: float = 1.0,
    reward_wrong: float = 0.0,
) -> list[float]:
    """
    Binary reward: 1.0 if extracted answer matches ground truth, else 0.0.
    Used as the primary reward in GRPO/RLVR.
    """
    rewards = []
    for completion, gold in zip(completions, ground_truths):
        pred = extract_answer(completion)
        gold_clean = extract_answer(gold) or gold
        correct = answers_equal(pred, gold_clean)
        rewards.append(reward_correct if correct else reward_wrong)
    return rewards


_THINK_BLOCK_RE = re.compile(
    r"<think>\s*(.*?)\s*</think>\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)


def format_reward(
    completions: Sequence[str],
    reward_valid: float = 0.1,
    reward_invalid: float = -0.1,
    require_answer_after_think: bool = True,
) -> list[float]:
    """
    Shaping reward for think-mode structure:
    - Exactly one <think>...</think> block
    - Non-empty content inside the block
    - Final answer follows after </think>
    """
    rewards = []
    for comp in completions:
        m = _THINK_BLOCK_RE.match(comp.strip())
        if m and m.group(1).strip() and m.group(2).strip():
            rewards.append(reward_valid)
        else:
            rewards.append(reward_invalid)
    return rewards


def language_consistency_reward(
    completions: Sequence[str],
    prompts: Sequence[str],
    target_language: str = "vie_Latn",
    min_confidence: float = 0.50,
    reward_consistent: float = 0.1,
    reward_inconsistent: float = -0.2,
    n_windows: int = 3,
    window_size: int = 100,
) -> list[float]:
    """
    For Vietnamese prompts: reward responses that reason in Vietnamese.
    Uses GlotLID / fastText to sample windows from the <think> trace.

    Implements the GreenMind language-consistency reward that prevents
    code-switching (model drifting to English/Chinese mid-reasoning).
    """
    rewards = []

    # Load lang-id model once
    try:
        import fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        lid_model = fasttext.load_model(model_path)
    except Exception:
        lid_model = None

    def _detect(text: str) -> tuple[str, float]:
        if lid_model is None:
            # Fallback heuristic
            vi_chars = set("ăâêôơưđĂÂÊÔƠƯĐ")
            ratio = sum(1 for c in text if c in vi_chars) / max(len(text), 1)
            if ratio > 0.01:
                return "vie_Latn", 0.70
            if any(c.isalpha() for c in text):
                return "eng_Latn", 0.60
            return "other", 0.50
        try:
            labels, scores = lid_model.predict(text.replace("\n", " ")[:512], k=1)
            return labels[0].replace("__label__", ""), float(scores[0])
        except Exception:
            return "other", 0.50

    def _is_vi_prompt(prompt: str) -> bool:
        lang, conf = _detect(prompt[:200])
        return ("vie" in lang or "vi" == lang) and conf >= min_confidence

    for completion, prompt in zip(completions, prompts):
        if not _is_vi_prompt(prompt):
            # Not a VI prompt; skip language reward (neutral)
            rewards.append(0.0)
            continue

        # Extract <think> trace
        m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
        trace = m.group(1) if m else completion

        if not trace.strip():
            rewards.append(reward_inconsistent)
            continue

        # Sample n_windows windows from the trace
        import random

        words = trace.split()
        votes_vi = 0
        for _ in range(n_windows):
            start = random.randint(0, max(0, len(words) - window_size))
            window = " ".join(words[start : start + window_size])
            lang, conf = _detect(window)
            if ("vie" in lang or "vi" == lang) and conf >= min_confidence:
                votes_vi += 1

        # Majority vote across windows
        if votes_vi >= n_windows // 2 + 1:
            rewards.append(reward_consistent)
        else:
            rewards.append(reward_inconsistent)

    return rewards


def combined_reward(
    completions: Sequence[str],
    ground_truths: Sequence[str],
    prompts: Sequence[str],
    weights: dict | None = None,
) -> list[float]:
    """
    Combine correctness + format + language_consistency rewards.
    weights: {"correctness": 1.0, "format": 0.1, "language": 0.15}
    """
    w = weights or {"correctness": 1.0, "format": 0.1, "language": 0.15}

    c_rewards = correctness_reward(completions, ground_truths)
    f_rewards = format_reward(completions)
    l_rewards = language_consistency_reward(completions, prompts)

    return [
        w["correctness"] * c + w["format"] * f + w["language"] * l
        for c, f, l in zip(c_rewards, f_rewards, l_rewards)
    ]
