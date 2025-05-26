from typing import Any, Dict, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import DynamicCache
# Ensure correct import path for your models
from .model import KVCompressor, CompressorConfig

class CacheProcessor(nn.Module):
    def __init__(self, config: CompressorConfig):
        super(CacheProcessor, self).__init__()
        self.config = config
        
        if config.kv_head_dim is None or config.kv_head_dim == 0:
            raise ValueError("CompressorConfig.kv_head_dim (number of KV heads in original cache) must be a positive integer.")
        self.model_num_kv_heads = config.kv_head_dim
        
        if config.input_dim % self.model_num_kv_heads != 0:
            raise ValueError(f"CompressorConfig.input_dim ({config.input_dim}) must be divisible by model_num_kv_heads ({self.model_num_kv_heads}).")
        self.model_head_dim_per_kv_head = config.input_dim // self.model_num_kv_heads

    def prepare_segments_for_compression(self, list_of_token_segments: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not list_of_token_segments:
            return None

        processed_segments = []
        for segment in list_of_token_segments:
            batch_size = segment.shape[0]
            segment_seq_len = segment.shape[2]
            
            reshaped_segment = segment.transpose(1, 2).contiguous().view(
                batch_size,
                segment_seq_len,
                self.config.input_dim 
            )
            processed_segments.append(reshaped_segment.unsqueeze(1)) 
        
        if not processed_segments: 
            return None
        concatenated_for_compressor = torch.cat(processed_segments, dim=1)
        return concatenated_for_compressor

    def format_compressed_segments_output(self, compressed_glob_tensor: torch.Tensor) -> List[torch.Tensor]:
        batch_size, num_segments_processed, compressed_segment_seq_len, _ = compressed_glob_tensor.shape
        
        output_segments_list = []
        for i in range(num_segments_processed):
            segment_data = compressed_glob_tensor[:, i, :, :] 
            
            reshaped_back = segment_data.view(
                batch_size,
                compressed_segment_seq_len,
                self.model_num_kv_heads,
                self.model_head_dim_per_kv_head
            )
            
            formatted_segment = reshaped_back.transpose(1, 2).contiguous()
            output_segments_list.append(formatted_segment)
            
        return output_segments_list


class CompCache(DynamicCache):
    def __init__(self, config: CompressorConfig, device: Optional[torch.device] = None):
        super().__init__() 
        self.config = config
        self._device = device 

        self.k_compressor = KVCompressor(config)
        self.v_compressor = KVCompressor(config)
        self._move_compressors_to_device_and_dtype()

        self.processor = CacheProcessor(config)

        self.key_cache: List[torch.Tensor] = [self._make_empty_layer_cache_placeholder() for _ in range(config.layer_nums)]
        self.value_cache: List[torch.Tensor] = [self._make_empty_layer_cache_placeholder() for _ in range(config.layer_nums)]
        self.compressed_seq_lengths: List[int] = [0] * config.layer_nums
        
    def _move_compressors_to_device_and_dtype(self):
        if self._device:
            self.k_compressor.to(self._device)
            self.v_compressor.to(self._device)
        self.k_compressor.to(dtype=self.config.torch_dtype)
        self.v_compressor.to(dtype=self.config.torch_dtype)

    def _ensure_device_from_input(self, input_tensor: torch.Tensor):
        # 确保压缩器和缓存实例在正确的设备上
        # 如果 _device 未设置，则从输入张量推断
        # 如果 _device 已设置但与输入张量不匹配，则更新 _device 并移动压缩器
        if self._device is None:
            self._device = input_tensor.device
            self._move_compressors_to_device_and_dtype()
        elif self._device != input_tensor.device: 
            # print(f"Warning: CompCache device changing from {self._device} to {input_tensor.device}") # 移除了警告
            self._device = input_tensor.device
            self._move_compressors_to_device_and_dtype()

    def _make_empty_layer_cache_placeholder(self) -> torch.Tensor:
        # 创建一个临时的1D空张量占位符。实际的4D形状在首次 update 时确定。
        target_device = self._device if self._device else torch.device("cpu")
        return torch.tensor([], dtype=self.config.torch_dtype, device=target_device)

    def update(
        self,
        key_states: torch.Tensor,   
        value_states: torch.Tensor, 
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self._ensure_device_from_input(key_states) # 确保设备一致性

        current_k_cache = self.key_cache[layer_idx]
        # 检查当前层缓存是否为有效的4D张量，或者维度是否与新输入匹配
        # 如果不是，则（重新）初始化该层的缓存为空的4D张量
        if current_k_cache.ndim != 4 or \
           current_k_cache.shape[0] != key_states.shape[0] or \
           current_k_cache.shape[1] != key_states.shape[1] or \
           current_k_cache.shape[3] != key_states.shape[3]:
            b, n, _, h = key_states.shape
            # 创建一个具有正确批次大小、头数、头维度但序列长度为0的空模板
            empty_template_k = torch.empty((b,n,0,h), dtype=key_states.dtype, device=key_states.device)
            empty_template_v = torch.empty((b,n,0,h), dtype=value_states.dtype, device=value_states.device)
            # 将新状态与空模板拼接，实际上是初始化了缓存
            self.key_cache[layer_idx] = torch.cat([empty_template_k, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([empty_template_v, value_states], dim=-2)
        else:
            # 如果缓存已正确初始化，则直接拼接新状态
            self.key_cache[layer_idx] = torch.cat([current_k_cache, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx == 0: # 通常 _seen_tokens 用于整个序列的计数
             self._seen_tokens += key_states.shape[-2]

        # 仅在处理完所有层之后尝试压缩
        if layer_idx == self.config.layer_nums - 1:
            self._attempt_compression()
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _get_layers_and_segments_for_compression(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[int]]:
        key_segments_to_compress = []
        value_segments_to_compress = []
        original_indices_of_segments = []
        current_compressed_len_in_zone_for_segments = [] 

        initial_keep_len = self.config.initial_uncompressed_keep_length

        for layer_idx in self.config.compress_layer_ids: 
            if layer_idx >= self.config.layer_nums: continue 
            # 必须是4D张量才能安全地获取形状
            if self.key_cache[layer_idx].ndim != 4: continue

            total_actual_len = self.key_cache[layer_idx].shape[-2]

            if total_actual_len <= initial_keep_len:
                continue

            len_of_compressed_output_in_zone = self.compressed_seq_lengths[layer_idx]
            start_of_pending_uncompressed_in_full_cache = initial_keep_len + len_of_compressed_output_in_zone
            length_of_pending_uncompressed = total_actual_len - start_of_pending_uncompressed_in_full_cache

            if length_of_pending_uncompressed >= self.config.compression_threshold:
                segment_extraction_start_idx = start_of_pending_uncompressed_in_full_cache
                segment_extraction_end_idx = segment_extraction_start_idx + self.config.compression_threshold
                
                key_segment = self.key_cache[layer_idx][:, :, segment_extraction_start_idx:segment_extraction_end_idx, :]
                value_segment = self.value_cache[layer_idx][:, :, segment_extraction_start_idx:segment_extraction_end_idx, :]
                
                if key_segment.shape[-2] == self.config.compression_threshold:
                    key_segments_to_compress.append(key_segment)
                    value_segments_to_compress.append(value_segment)
                    original_indices_of_segments.append(layer_idx)
                    current_compressed_len_in_zone_for_segments.append(len_of_compressed_output_in_zone)
        
        return key_segments_to_compress, value_segments_to_compress, original_indices_of_segments, current_compressed_len_in_zone_for_segments

    def _attempt_compression(self):
        key_segments, value_segments, segment_original_indices, segment_old_compressed_len_in_zone = \
            self._get_layers_and_segments_for_compression()

        if not segment_original_indices: 
            return

        # 确保压缩器在正确的设备上 (如果未设置，则从数据推断)
        if key_segments: self._ensure_device_from_input(key_segments[0])
        elif value_segments: self._ensure_device_from_input(value_segments[0])
        else: return # 如果没有可压缩的段，则返回

        initial_keep_len = self.config.initial_uncompressed_keep_length

        newly_compressed_k_segments = []
        if key_segments:
            glob_k_input_for_compressor = self.processor.prepare_segments_for_compression(key_segments)
            if glob_k_input_for_compressor is not None and glob_k_input_for_compressor.numel() > 0:
                compressed_k_glob = self.k_compressor(glob_k_input_for_compressor)
                newly_compressed_k_segments = self.processor.format_compressed_segments_output(compressed_k_glob)

        newly_compressed_v_segments = []
        if value_segments:
            glob_v_input_for_compressor = self.processor.prepare_segments_for_compression(value_segments)
            if glob_v_input_for_compressor is not None and glob_v_input_for_compressor.numel() > 0:
                compressed_v_glob = self.v_compressor(glob_v_input_for_compressor)
                newly_compressed_v_segments = self.processor.format_compressed_segments_output(compressed_v_glob)

        for i, original_layer_idx in enumerate(segment_original_indices):
            current_segment_old_compressed_output_len_in_zone = segment_old_compressed_len_in_zone[i]

            # --- 处理键缓存 ---
            if i < len(newly_compressed_k_segments):
                new_k_seg = newly_compressed_k_segments[i] 
                
                # 假设此时 self.key_cache[original_layer_idx] 是有效的4D张量
                # 如果不是，之前的逻辑应该已经处理或会引发错误
                initial_keep_k_part = self.key_cache[original_layer_idx][:, :, :initial_keep_len, :]
                old_compressed_k_part_in_zone = self.key_cache[original_layer_idx][:, :, 
                    initial_keep_len : initial_keep_len + current_segment_old_compressed_output_len_in_zone, 
                :]
                start_of_trailing_uncompressed_k = initial_keep_len + \
                                                   current_segment_old_compressed_output_len_in_zone + \
                                                   self.config.compression_threshold
                uncompressed_trailing_k_part = self.key_cache[original_layer_idx][:, :, start_of_trailing_uncompressed_k:, :]
                
                self.key_cache[original_layer_idx] = torch.cat([
                    initial_keep_k_part, 
                    old_compressed_k_part_in_zone, 
                    new_k_seg, 
                    uncompressed_trailing_k_part
                ], dim=-2)

            # --- 处理值缓存 ---
            if i < len(newly_compressed_v_segments):
                new_v_seg = newly_compressed_v_segments[i]

                initial_keep_v_part = self.value_cache[original_layer_idx][:, :, :initial_keep_len, :]
                old_compressed_v_part_in_zone = self.value_cache[original_layer_idx][:, :, 
                    initial_keep_len : initial_keep_len + current_segment_old_compressed_output_len_in_zone, 
                :]
                start_of_trailing_uncompressed_v = initial_keep_len + \
                                                   current_segment_old_compressed_output_len_in_zone + \
                                                   self.config.compression_threshold
                uncompressed_trailing_v_part = self.value_cache[original_layer_idx][:, :, start_of_trailing_uncompressed_v:, :]
                
                self.value_cache[original_layer_idx] = torch.cat([
                    initial_keep_v_part, 
                    old_compressed_v_part_in_zone, 
                    new_v_seg, 
                    uncompressed_trailing_v_part
                ], dim=-2)
                
                self.compressed_seq_lengths[original_layer_idx] = current_segment_old_compressed_output_len_in_zone + new_v_seg.shape[-2]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # 确保在访问 shape 之前张量是有效的4D张量且非空
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx].ndim != 4 or not self.key_cache[layer_idx].numel():
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        # 在 beam search 时重新排序缓存
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].ndim == 4 and self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            if self.value_cache[layer_idx].ndim == 4 and self.value_cache[layer_idx].numel() > 0:
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        # 转换为 Hugging Face期望的元组格式
        return tuple(self.key_cache), tuple(self.value_cache)

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]], config: CompressorConfig, device: Optional[torch.device] = None) -> "CompCache":
        # 从旧格式的 past_key_values 创建 CompCache 实例
        cache = cls(config, device)
        if past_key_values is None or len(past_key_values) == 0 or \
           not past_key_values[0] or len(past_key_values[0]) == 0 : # 进一步检查内部元组是否也为空
            return cache
            
        key_tensors_tuple, value_tensors_tuple = past_key_values
        
        num_layers_in_legacy = len(key_tensors_tuple)
        
        # 使用占位符初始化，然后填充有效数据
        temp_key_cache = [cache._make_empty_layer_cache_placeholder() for _ in range(config.layer_nums)]
        temp_value_cache = [cache._make_empty_layer_cache_placeholder() for _ in range(config.layer_nums)]

        first_valid_tensor = None

        for i in range(min(num_layers_in_legacy, config.layer_nums)):
            if key_tensors_tuple[i] is not None and key_tensors_tuple[i].numel() > 0:
                temp_key_cache[i] = key_tensors_tuple[i]
                if first_valid_tensor is None: first_valid_tensor = key_tensors_tuple[i]
            if value_tensors_tuple[i] is not None and value_tensors_tuple[i].numel() > 0:
                temp_value_cache[i] = value_tensors_tuple[i]
                if first_valid_tensor is None: first_valid_tensor = value_tensors_tuple[i]

        cache.key_cache = temp_key_cache
        cache.value_cache = temp_value_cache
        
        cache.compressed_seq_lengths = [0] * config.layer_nums 
        if first_valid_tensor is not None:
            cache._ensure_device_from_input(first_valid_tensor)
            cache._seen_tokens = first_valid_tensor.shape[-2] 
        return cache
