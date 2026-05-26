import torch

from models.configuration_ouro import OuroConfig
from models.modeling_ouro import OuroForCausalLM
from models.looplm_train import LoopLMLossConfig, OuroLoopLMTrain


def test_looplm_loss_forward():
    torch.manual_seed(0)
    config = OuroConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        total_ut_steps=3,
    )
    model = OuroForCausalLM(config)
    trainer = OuroLoopLMTrain(
        model=model,
        config=LoopLMLossConfig(kl_beta=0.1, include_adaptive_exit_loss=True, adaptive_exit_weight=0.05),
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.roll(-1, dims=1)
    labels[:, -1] = -100
    out = trainer(input_ids=input_ids, labels=labels, total_ut_steps=3)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert out["mean_exit_step"] >= 1.0
    assert out["mean_exit_step"] <= 3.0 + 1e-4

