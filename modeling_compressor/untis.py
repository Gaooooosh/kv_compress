from dataclasses import dataclass
from typing import Any
@dataclass
class CompressorConfig:
    input_dim: int = 4096
    reduction_factor: int = 16
    output_seq_len: int = 2
    num_attention_heads: int = 4
    use_mixed_precision: bool = True
    kv_head_dim: int = 4
    layer_nums:int = 32
    compress_layers:list[int] = None
    torch_dtype: Any = None