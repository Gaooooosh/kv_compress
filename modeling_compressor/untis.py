from dataclasses import dataclass
from typing import Any, List, Optional
@dataclass
# In your model.py or untis.py where CompressorConfig is defined
class CompressorConfig:
    def __init__(self, input_dim, reduction_factor, output_seq_len,
                 num_attention_heads, use_mixed_precision, torch_dtype,
                 kv_head_dim, layer_nums, # Existing fields from your infra.py
                 kernel_size=3, padding=1, # Sensible defaults
                 # New fields for incremental and selective compression
                 compression_threshold: int = 16, # Default threshold
                 compress_layer_ids: Optional[List[int]] = None, # Default to all layers
                 initial_uncompressed_keep_length: int = 0
                ):
        self.input_dim = input_dim
        self.reduction_factor = reduction_factor
        self.output_seq_len = output_seq_len # Target sequence length after compression for a segment
        self.num_attention_heads = num_attention_heads # For KVCompressor's internal attention
        self.use_mixed_precision = use_mixed_precision
        self.torch_dtype = torch_dtype
        
        self.kv_head_dim = kv_head_dim # In your setup, this seems to be num_kv_heads for the main model
        self.layer_nums = layer_nums # Total layers in the main model
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.initial_uncompressed_keep_length = initial_uncompressed_keep_length
        self.compression_threshold = compression_threshold
        self.compress_layer_ids = compress_layer_ids if compress_layer_ids is not None else list(range(layer_nums))
        # If compress_layer_ids is None, default to all layers.