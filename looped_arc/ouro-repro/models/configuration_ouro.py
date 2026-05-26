from dataclasses import dataclass


@dataclass
class OuroConfig:
    vocab_size: int = 49152
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    intermediate_size: int = 5632
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    total_ut_steps: int = 4
    early_exit_threshold: float = 1.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True

