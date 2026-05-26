from .configuration_ouro import OuroConfig


def ouro_1_4b_config() -> OuroConfig:
    return OuroConfig(
        vocab_size=49152,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        intermediate_size=5632,
        total_ut_steps=4,
        max_position_embeddings=32768,
        rope_theta=10000.0,
    )


def ouro_2_6b_config() -> OuroConfig:
    # Paper reports 2.6B via upcycling; this preset keeps width fixed
    # and increases depth as a practical open approximation.
    return OuroConfig(
        vocab_size=49152,
        hidden_size=2048,
        num_hidden_layers=48,
        num_attention_heads=32,
        intermediate_size=5632,
        total_ut_steps=4,
        max_position_embeddings=32768,
        rope_theta=10000.0,
    )

