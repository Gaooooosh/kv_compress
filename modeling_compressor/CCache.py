from typing import Any, Dict, Iterable, Optional, Tuple
import torch
import torch.nn as nn
from transformers import DynamicCache
from .model import KVCompressor,CompressorConfig

class CacheProcessor(nn.Module):
    def __init__(self, config: CompressorConfig):
        super(CacheProcessor, self).__init__()
        self.kv_head_dim = config.kv_head_dim
        self.to_length = config.output_seq_len
        self.config = config
    def forward(self, cache: list):
        # Check cache structure validity
        batch_size = cache[0].size(0)
        # Concatenate caches from each layer
        layers_num = len(cache)
        seq_len = cache[0][0].size(1)
        # Concatenate heads and head_dim: (batch_size, layers * num_heads, seq_len, head_dim)
        combined_cache = torch.cat([v.transpose(1,2).unsqueeze(1) for v in cache], dim=1)  # Concatenates on num_heads dimension
        # Reshape to merge num_heads and head_dim: (batch_size, layers_num, seq_len, input_dim)
        processed_cache = combined_cache.view(batch_size, layers_num, seq_len, -1)
        return processed_cache

    def compressed_tensor_to_cache(self,k,v):
        batch_size, layer_num, seq_len, hidden_dim = k.size()
        k = k.view(batch_size, layer_num, self.to_length, self.kv_head_dim, -1).transpose(2,3)
        v = v.view(batch_size, layer_num, self.to_length, self.kv_head_dim, -1).transpose(2,3)
        return CompCache(self.config,[(k[:,i,:,:,:],v[:,i,:,:,:]) for i in range(layer_num)])

    def compressed_tensor_to_list(self,t):
        batch_size, layer_num, seq_len, hidden_dim = t.size()
        t = t.view(batch_size, layer_num, self.to_length, self.kv_head_dim, -1).transpose(2,3)
        return [t[:,i,:,:,:] for i in range(layer_num)]


class CompCache(DynamicCache):
    def __init__(self, config:CompressorConfig,_distributed_cache_data: Iterable = None):
        super().__init__(_distributed_cache_data)
        self.k_compressor = KVCompressor(config)
        self.v_compressor = KVCompressor(config)
        self.processor = CacheProcessor(config)
        self.compressed_token = 0
        self.layer_nums = config.layer_nums
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx == self.layer_nums - 1:
            self.compress()
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def compress(self):
        self.key_cache = self.k_compressor(self.processor(self.key_cache))
        self.value_cache = self.v_compressor(self.processor(self.value_cache))
        