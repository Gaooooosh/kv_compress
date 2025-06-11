# CCache.py (Simplified for Global Multi-Layer Compression)
from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import DynamicCache

from untils import CompressorConfig
from model import KVCompressor

class CacheProcessor(nn.Module):
    def __init__(self, config: CompressorConfig):
        super(CacheProcessor, self).__init__()
        self.config = config
        
        if config.kv_head_dim is None or config.kv_head_dim == 0:
            raise ValueError("CompressorConfig.kv_head_dim must be a positive integer.")
        self.model_num_kv_heads = config.kv_head_dim
        
        if not hasattr(config, 'input_dim') or config.input_dim is None:
             raise ValueError("CompressorConfig.input_dim must be set.")
        if config.input_dim % self.model_num_kv_heads != 0:
            raise ValueError(
                f"CompressorConfig.input_dim ({config.input_dim}) must be divisible "
                f"by model_num_kv_heads ({self.model_num_kv_heads})."
            )
        self.model_head_dim_per_kv_head = config.input_dim // self.model_num_kv_heads

    def prepare_segments_for_global_compression(self,
                                                list_of_layer_segments: List[torch.Tensor],
                                                expected_segment_seq_len: int
                                                ) -> Optional[torch.Tensor]:
        """
        将来自多个层的、每个长度为 expected_segment_seq_len 的段，准备成一个给KVCompressor的输入。
        KVCompressor的forward方法期望的输入x形状: (batch_size, l_layers, seq_len_orig, dim_features)
        这里，l_layers 就是 list_of_layer_segments 的长度。
        """
        if not list_of_layer_segments:
            return None
        
        batch_size_example = list_of_layer_segments[0].shape[0]
        num_actual_layers_to_compress = len(list_of_layer_segments)

        # 将每个 (B, H, S_seg, D_head) 的段转换为 (B, S_seg, input_dim)
        reshaped_segments_for_stacking = []
        for segment in list_of_layer_segments: # segment: (B, H, S_seg, D_head)
            if segment.shape[2] != expected_segment_seq_len:
                # 如果实际段长度与期望不符，可能需要填充或报错
                # 为简单起见，这里假设长度都符合
                print(f"Warning: Segment seq len {segment.shape[2]} != expected {expected_segment_seq_len}")
                # Pad or truncate if necessary, or raise error. For now, assume they match.
            
            batch_size = segment.shape[0]
            # (B, S_seg, H * D_head = input_dim)
            reshaped_segment = segment.transpose(1, 2).contiguous().view(
                batch_size,
                segment.shape[2], 
                self.config.input_dim 
            )
            reshaped_segments_for_stacking.append(reshaped_segment)
        
        # 将这些 (B, S_seg, input_dim) 的张量在新的“层”维度上堆叠
        # 得到 (B, num_actual_layers_to_compress, S_seg, input_dim)
        if not reshaped_segments_for_stacking:
            return None
        
        # 确保所有batch_size一致
        # batch_size_example = reshaped_segments_for_stacking[0].shape[0]
        # stacked_tensor = torch.stack(reshaped_segments_for_stacking, dim=1)
        # return stacked_tensor
        
        # 修正：KVCompressor的forward是 (B*L, S, D) -> permute -> (B*L, D, S) for conv1d
        # 然后 view 回 (B, L, S_reduced, D) for pooling
        # 不，KVCompressor的forward接收 (B, L, S_orig, D_feat)
        # 所以上面的 stack 应该是对的
        try:
            stacked_tensor = torch.stack(reshaped_segments_for_stacking, dim=1)
            return stacked_tensor # Shape: (B, num_actual_layers_to_compress, expected_segment_seq_len, input_dim)
        except RuntimeError as e:
            print(f"Error stacking segments in CacheProcessor: {e}")
            for i, seg_rs in enumerate(reshaped_segments_for_stacking):
                print(f"  Reshaped segment {i} shape: {seg_rs.shape}")
            return None


    def format_globally_compressed_output(self, 
                                          globally_compressed_tensor: torch.Tensor,
                                          num_original_layers_compressed: int
                                          ) -> List[torch.Tensor]:
        """
        将KVCompressor输出的全局压缩张量，切分回对应数量的层。
        globally_compressed_tensor shape: (B, num_original_layers_compressed, S_compressed_new, input_dim)
        返回一个List[Tensor]，每个Tensor是 (B, H, S_compressed_new, D_head)
        """
        batch_size, num_layers_in_output, compressed_segment_seq_len, _ = globally_compressed_tensor.shape

        if num_layers_in_output != num_original_layers_compressed:
            # 这通常不应该发生，如果KVCompressor的 l_layers 维度保持不变
            print(f"Warning: Num layers in compressed output ({num_layers_in_output}) "
                  f"differs from num original layers compressed ({num_original_layers_compressed}).")
            # 可能需要调整或报错

        output_segments_list = []
        for i in range(num_layers_in_output): # 遍历压缩输出中的每一“层”
            layer_compressed_data = globally_compressed_tensor[:, i, :, :] # (B, S_compressed_new, input_dim)
            
            # Reshape back to (B, S_compressed_new, num_kv_heads, head_dim_per_kv_head)
            reshaped_back = layer_compressed_data.view(
                batch_size,
                compressed_segment_seq_len,
                self.model_num_kv_heads,
                self.model_head_dim_per_kv_head
            )
            # Transpose to (B, num_kv_heads, S_compressed_new, head_dim_per_kv_head)
            formatted_segment = reshaped_back.transpose(1, 2).contiguous()
            output_segments_list.append(formatted_segment)
            
        return output_segments_list


class CompCache(DynamicCache):
    def __init__(self, config: CompressorConfig, device: Optional[torch.device] = None):
        super().__init__() 
        self.config = config
        self._device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        if isinstance(self.config.torch_dtype, str):
            dtype_map = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16, "torch.float32": torch.float32}
            self.config.torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float32)
        elif not isinstance(self.config.torch_dtype, torch.dtype):
             self.config.torch_dtype = torch.float32

        self.k_compressor = KVCompressor(config)
        self.v_compressor = KVCompressor(config)
        self._move_compressors_to_device_and_dtype()
        self.processor = CacheProcessor(config)

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._initialize_cache_lists()
        self._seen_tokens = 0

    def _initialize_cache_lists(self, batch_size: Optional[int] = None, num_heads: Optional[int] = None, head_dim: Optional[int] = None):
        self.key_cache = []
        self.value_cache = []
        for _ in range(self.config.layer_nums):
            if batch_size is not None and num_heads is not None and head_dim is not None:
                empty_tensor = torch.empty((batch_size, num_heads, 0, head_dim), dtype=self.config.torch_dtype, device=self._device)
                self.key_cache.append(empty_tensor.clone()) # Use clone for safety
                self.value_cache.append(empty_tensor.clone())
            else:
                self.key_cache.append(self._make_empty_layer_cache_placeholder(self.config.torch_dtype))
                self.value_cache.append(self._make_empty_layer_cache_placeholder(self.config.torch_dtype))
        self._seen_tokens = 0

    def _move_compressors_to_device_and_dtype(self): # (保持不变)
        if self._device:
            self.k_compressor.to(self._device)
            self.v_compressor.to(self._device)
        target_dtype = self.config.torch_dtype
        if target_dtype == torch.bfloat16 and self._device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            target_dtype = torch.float32
        self.k_compressor.to(dtype=target_dtype)
        self.v_compressor.to(dtype=target_dtype)

    def _ensure_device_from_input(self, input_tensor: torch.Tensor): # (保持不变)
        new_device = input_tensor.device
        if self._device != new_device: 
            self._device = new_device
            self._move_compressors_to_device_and_dtype()

    def _make_empty_layer_cache_placeholder(self, dtype: torch.dtype) -> torch.Tensor: # (保持不变)
        return torch.tensor([], dtype=dtype, device=self._device)

    def _ensure_layer_cache_initialized(self, layer_idx: int, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype): # (保持不变)
        while len(self.key_cache) <= layer_idx: self.key_cache.append(self._make_empty_layer_cache_placeholder(dtype))
        while len(self.value_cache) <= layer_idx: self.value_cache.append(self._make_empty_layer_cache_placeholder(dtype))
        current_k = self.key_cache[layer_idx]
        if not (current_k.ndim == 4 and current_k.shape[0] == batch_size and \
                current_k.shape[1] == num_heads and current_k.shape[3] == head_dim):
            self.key_cache[layer_idx] = torch.empty((batch_size, num_heads, 0, head_dim), dtype=dtype, device=self._device)
        current_v = self.value_cache[layer_idx]
        if not (current_v.ndim == 4 and current_v.shape[0] == batch_size and \
                current_v.shape[1] == num_heads and current_v.shape[3] == head_dim):
            self.value_cache[layer_idx] = torch.empty((batch_size, num_heads, 0, head_dim), dtype=dtype, device=self._device)

    def update( # (基本保持不变，负责追加增量)
        self,
        key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_device_from_input(key_states) 
        batch_size, num_heads, incremental_seq_len, head_dim = key_states.shape
        self._ensure_layer_cache_initialized(layer_idx, batch_size, num_heads, head_dim, key_states.dtype)

        # 当拼接时，确保旧的缓存部分是分离的
        # key_states 和 value_states (增量) 是新的，带有当前的计算图
        old_k_cache = self.key_cache[layer_idx].detach().clone() # 分离并复制旧的缓存内容
        old_v_cache = self.value_cache[layer_idx].detach().clone() # 分离并复制旧的缓存内容

        self.key_cache[layer_idx] = torch.cat([old_k_cache, key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([old_v_cache, value_states], dim=-2)

        if layer_idx == 0: self._seen_tokens += incremental_seq_len
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_from_hf_past( # (基本保持不变，负责重置)
        self, hf_past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if not hf_past_key_values or not hf_past_key_values[0] or not hf_past_key_values[0][0].numel():
            self._initialize_cache_lists()
            return
        first_key_tensor = hf_past_key_values[0][0]
        self._ensure_device_from_input(first_key_tensor)
        batch_size, num_kv_heads, current_hf_total_seq_len, head_dim = first_key_tensor.shape
        self._initialize_cache_lists(batch_size, num_kv_heads, head_dim)
        self._seen_tokens = current_hf_total_seq_len
        for layer_idx in range(min(len(hf_past_key_values), self.config.layer_nums)):
            new_hf_k, new_hf_v = hf_past_key_values[layer_idx]
            if new_hf_k.ndim == 4 and new_hf_v.ndim == 4:
                 self.key_cache[layer_idx] = new_hf_k.clone()
                 self.value_cache[layer_idx] = new_hf_v.clone()

    def trigger_global_compression(self) -> bool:
        """
        对配置中指定的所有层（通常跳过前几层），提取固定长度的段进行全局压缩。
        """
        actual_compression_performed = False
        initial_keep_len = self.config.initial_uncompressed_keep_length
        segment_len_to_compress = self.config.compression_threshold # 这是每层要提取的段长

        segments_k_to_compress_globally = []
        segments_v_to_compress_globally = []
        
        # 记录哪些原始层参与了这次全局压缩，及其原始索引
        participating_layer_indices = [] 

        # 确保压缩器设备正确
        if len(self.config.compress_layer_ids) > 0 and \
           len(self.key_cache) > self.config.compress_layer_ids[0] and \
           self.key_cache[self.config.compress_layer_ids[0]].numel() > 0 :
             self._ensure_device_from_input(self.key_cache[self.config.compress_layer_ids[0]])
        
        # 1. 从每个指定层收集可压缩段
        for layer_idx in self.config.compress_layer_ids:
            if layer_idx >= len(self.key_cache): continue

            current_k_layer = self.key_cache[layer_idx]
            current_v_layer = self.value_cache[layer_idx]

            if not (current_k_layer.ndim == 4 and current_k_layer.numel() > 0):
                continue 

            current_total_len = current_k_layer.shape[-2]
            compressible_zone_len = current_total_len - initial_keep_len
            
            if compressible_zone_len >= segment_len_to_compress:
                segment_start_idx = initial_keep_len
                segment_end_idx = initial_keep_len + segment_len_to_compress
                
                key_segment = current_k_layer[:, :, segment_start_idx:segment_end_idx, :]
                value_segment = current_v_layer[:, :, segment_start_idx:segment_end_idx, :]

                if key_segment.shape[-2] == segment_len_to_compress: # 确保提取了正确长度
                    segments_k_to_compress_globally.append(key_segment)
                    segments_v_to_compress_globally.append(value_segment)
                    participating_layer_indices.append(layer_idx)
                # else:
                    # print(f"DEBUG Global: Layer {layer_idx} could not extract full segment of length {segment_len_to_compress}")
            # else:
                # print(f"DEBUG Global: Layer {layer_idx} not enough len ({compressible_zone_len}) for threshold {segment_len_to_compress}")


        if not participating_layer_indices: # 没有从任何层收集到足够的段
            # print("DEBUG Global: No participating layers for compression.")
            return False

        # 2. 将收集到的所有段准备并送入压缩器
        # prepare_segments_for_global_compression 返回 (B, num_participating_layers, S_seg, D_input)
        prepared_k_global = self.processor.prepare_segments_for_global_compression(
            segments_k_to_compress_globally, segment_len_to_compress
        )
        prepared_v_global = self.processor.prepare_segments_for_global_compression(
            segments_v_to_compress_globally, segment_len_to_compress
        )

        if prepared_k_global is None or prepared_v_global is None or \
           prepared_k_global.numel() == 0 or prepared_v_global.numel() == 0:
            # print("DEBUG Global: Prepared global segments are empty. No compression performed.")
            return False # 准备阶段失败或无数据

        # 3. 执行全局压缩
        # KVCompressor.forward 接收 (B, L, S_orig, D_feat), L是参与压缩的层数
        # 输出 (B, L, S_compressed_new, D_feat)
        globally_compressed_k = self.k_compressor(prepared_k_global)
        globally_compressed_v = self.v_compressor(prepared_v_global)

        # 4. 将压缩后的全局结果切分并更新回各个原始层
        # format_globally_compressed_output 返回 List[Tensor], 每个 (B, H, S_comp_new, D_head)
        list_of_new_k_segments = self.processor.format_globally_compressed_output(
            globally_compressed_k, len(participating_layer_indices)
        )
        list_of_new_v_segments = self.processor.format_globally_compressed_output(
            globally_compressed_v, len(participating_layer_indices)
        )

        if not list_of_new_k_segments or not list_of_new_v_segments or \
           len(list_of_new_k_segments) != len(participating_layer_indices) or \
           list_of_new_k_segments[0].shape[-2] == 0: # 检查压缩器是否有有效输出
            print("DEBUG Global: Compressor output is empty or layer count mismatch. Effective segment removal.")
            # 即使压缩输出为空，我们也要从原始层中移除被“消耗”的段
            for idx, original_layer_idx in enumerate(participating_layer_indices):
                current_k_layer = self.key_cache[original_layer_idx]
                current_v_layer = self.value_cache[original_layer_idx]
                
                segment_end_idx = initial_keep_len + segment_len_to_compress
                head_k_part = current_k_layer[:, :, :initial_keep_len, :].detach().clone() # Detach 历史部分
                tail_k_part = current_k_layer[:, :, segment_end_idx:, :].detach().clone() # Detach 历史部分
                self.key_cache[original_layer_idx] = torch.cat([head_k_part, tail_k_part], dim=-2)
                
                head_v_part = current_v_layer[:, :, :initial_keep_len, :].detach().clone() # Detach 历史部分
                tail_v_part = current_v_layer[:, :, segment_end_idx:, :].detach().clone() # Detach 历史部分
                self.value_cache[original_layer_idx] = torch.cat([head_v_part, tail_v_part], dim=-2)
            actual_compression_performed = True # 认为进行了一次操作（即使是删除）
            return actual_compression_performed


        actual_compression_performed = True
        for i, original_layer_idx in enumerate(participating_layer_indices):
            new_k_seg_for_layer = list_of_new_k_segments[i]
            new_v_seg_for_layer = list_of_new_v_segments[i]

            current_k_layer = self.key_cache[original_layer_idx] # 这是压缩操作前的状态
            current_v_layer = self.value_cache[original_layer_idx]
            head_k_part = current_k_layer[:, :, :initial_keep_len, :]
            head_v_part = current_v_layer[:, :, :initial_keep_len, :]
            
            segment_end_idx_original = initial_keep_len + segment_len_to_compress
            tail_k_part = current_k_layer[:, :, segment_end_idx_original:, :]
            tail_v_part = current_v_layer[:, :, segment_end_idx_original:, :]
            
            self.key_cache[original_layer_idx] = torch.cat([head_k_part, new_k_seg_for_layer, tail_k_part], dim=-2)
            self.value_cache[original_layer_idx] = torch.cat([head_v_part, new_v_seg_for_layer, tail_v_part], dim=-2)

        return actual_compression_performed

    # --- DynamicCache API (与上一版简化版基本一致) ---
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int: # (保持不变)
        if layer_idx is None: layer_idx = 0
        if layer_idx >= len(self.key_cache) or not (self.key_cache[layer_idx].ndim == 4):
            return 0 
        return self.key_cache[layer_idx].shape[-2] 

    def get_max_length(self) -> Optional[int]: return None  # (保持不变)

    def get_usable_length(self, new_sequence_length: Optional[int]=None, layer_idx: Optional[int] = 0) -> int: # (保持不变)
        return self._seen_tokens 

    def reorder_cache(self, beam_idx: torch.LongTensor): # (保持不变)
        for layer_idx in range(len(self.key_cache)): 
            if self.key_cache[layer_idx].ndim == 4 and self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            if self.value_cache[layer_idx].ndim == 4 and self.value_cache[layer_idx].numel() > 0:
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]: # (保持不变)
        return tuple(self.key_cache[:self.config.layer_nums]), tuple(self.value_cache[:self.config.layer_nums])

    @classmethod
    def from_legacy_cache( # (保持不变)
        cls, 
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None, 
        compressor_config: Optional[CompressorConfig] = None,
    ) -> "CompCache":
        if compressor_config is None:
            raise ValueError("CompressorConfig must be provided to create CompCache from legacy cache.")
        cache = cls(compressor_config) 
        if past_key_values is None or len(past_key_values) == 0: return cache
        cache.update_from_hf_past(past_key_values)
        return cache


if __name__ == '__main__':
    print("--- Running Simplified Test for CompCache (Global Multi-Layer Compression) ---")
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
    try:
        from config_rl import DEFAULT_COMPRESSOR_CONFIG_PARAMS, LLM_CONFIG
        use_config_file = True
    except ImportError:
        print("Warning: config_rl.py not found or params missing. Using hardcoded test config.")
        DEFAULT_COMPRESSOR_CONFIG_PARAMS = {
            "reduction_factor": 2, "output_seq_len": 4, "num_attention_heads": 2,
            "use_mixed_precision": False, "torch_dtype": "torch.float32",
            "kernel_size": 3, "padding": 1,
            "compression_threshold": 8, "initial_uncompressed_keep_length": 2, # 每层提取8个token
            "layer_nums": 4, "kv_head_dim": 2, "input_dim": 128 # 假设LLM有4层
        }
        LLM_CONFIG = {"model_name_or_path": None} 
        use_config_file = False

    test_layer_nums = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("layer_nums", 4)
    test_kv_heads = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kv_head_dim", 2)
    test_input_dim = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("input_dim", 128)

    if use_config_file and LLM_CONFIG.get("model_name_or_path"):
        try:
            from transformers import AutoConfig
            llm_conf = AutoConfig.from_pretrained(LLM_CONFIG["model_name_or_path"])
            test_layer_nums = llm_conf.num_hidden_layers
            test_kv_heads = getattr(llm_conf, 'num_key_value_heads', llm_conf.num_attention_heads)
            test_input_dim = (llm_conf.hidden_size // llm_conf.num_attention_heads) * test_kv_heads
            print(f"Auto-detected for test: layer_nums={test_layer_nums}, kv_heads={test_kv_heads}, input_dim={test_input_dim}")
        except Exception as e:
            print(f"Could not auto-detect LLM params, using defaults: {e}")

    # 通常跳过前几层，例如跳过前4层。如果总共只有4层，那就压缩0层了。
    # 为了测试，我们假设总共有6层，跳过前2层，压缩后面4层。
    # test_layer_nums = 6
    # compress_layer_ids_for_test = list(range(2, 6)) # 压缩层 2, 3, 4, 5
    # 如果总层数少，比如4层，跳过2层，压缩2层
    if test_layer_nums > 4:
        skip_layers = 4
    elif test_layer_nums > 2 :
        skip_layers = 2
    else:
        skip_layers = 0
    
    compress_layer_ids_for_test = list(range(skip_layers, test_layer_nums))
    if not compress_layer_ids_for_test and test_layer_nums > 0 : # 如果跳过后没层了但总层数>0，则至少压缩最后一层
        compress_layer_ids_for_test = [test_layer_nums -1]
    elif not compress_layer_ids_for_test and test_layer_nums == 0: # 如果总层数为0
        compress_layer_ids_for_test = []


    print(f"Test Config: Total LLM Layers={test_layer_nums}, Layers to Compress IDs={compress_layer_ids_for_test}")

    test_config = CompressorConfig(
        input_dim=test_input_dim,
        reduction_factor=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("reduction_factor", 2),
        output_seq_len=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("output_seq_len", 4), # 压缩后每“逻辑层段”的长度
        num_attention_heads=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("num_attention_heads", 2),
        use_mixed_precision=False,
        torch_dtype=torch.float32,
        kv_head_dim=test_kv_heads,
        layer_nums=test_layer_nums, # KVCompressor的config中的layer_nums现在指它一次处理多少个拼接来的物理层
        kernel_size=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kernel_size", 3),
        padding=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("padding", 1),
        compression_threshold=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("compression_threshold", 4), # 每层提取的未压缩段长度
        initial_uncompressed_keep_length=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("initial_uncompressed_keep_length", 4),
        compress_layer_ids=compress_layer_ids_for_test 
    )
    # 重要：调整KVCompressor的config.layer_nums以匹配实际一次送入的层数
    # test_config.layer_nums = len(compress_layer_ids_for_test) # 这一步应该在KVCompressor内部处理，它看到的是 (B,L,S,D)

    device = torch.device("cpu") 
    print(f"Test running on device: {device}")
    
    comp_cache_instance = CompCache(test_config, device=device)

    batch_size = 1
    num_kv_heads_for_test = test_config.kv_head_dim
    head_dim_for_test = test_config.input_dim // num_kv_heads_for_test
    dtype_for_test = test_config.torch_dtype
    print(f"Test params: B={batch_size}, H_kv={num_kv_heads_for_test}, D_head={head_dim_for_test}")

    def _generate_kv_increment(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k = torch.randn(batch_size, num_kv_heads_for_test, seq_len, head_dim_for_test, dtype=dtype_for_test, device=device)
        v = torch.randn(batch_size, num_kv_heads_for_test, seq_len, head_dim_for_test, dtype=dtype_for_test, device=device)
        return k, v

    print("\nSimulating 'update' calls...")
    tokens_to_add_for_compression = test_config.initial_uncompressed_keep_length + test_config.compression_threshold # 2 + 8 = 10
    
    print(f"  Adding {tokens_to_add_for_compression} tokens to all layers to make them compressible...")
    for layer_idx in range(test_config.layer_nums): # 给所有LLM层添加数据
        k_inc, v_inc = _generate_kv_increment(tokens_to_add_for_compression)
        comp_cache_instance.update(k_inc, v_inc, layer_idx)
    
    print(f"  After adding {tokens_to_add_for_compression} tokens (before compression attempt):")
    for layer_idx in range(test_config.layer_nums):
        if layer_idx < 2 or layer_idx in compress_layer_ids_for_test : # 打印前几层和要压缩的层
            print(f"    Layer {layer_idx}: seq_len={comp_cache_instance.get_seq_length(layer_idx)}")
    print(f"    _seen_tokens: {comp_cache_instance._seen_tokens}")

    print("\n--- Attempting Global Compression ---")
    performed = comp_cache_instance.trigger_global_compression()
    print(f"  Global compression performed: {performed}")
    
    expected_len_after_compress = test_config.initial_uncompressed_keep_length + \
                                  test_config.output_seq_len + \
                                  (tokens_to_add_for_compression - test_config.initial_uncompressed_keep_length - test_config.compression_threshold)
    # = 2 + 4 + (10 - 2 - 8) = 6
    
    if performed:
        for layer_idx in range(test_config.layer_nums):
            if layer_idx < 2 or layer_idx in compress_layer_ids_for_test :
                print(f"    Layer {layer_idx}: new_seq_len={comp_cache_instance.get_seq_length(layer_idx)}")
                if layer_idx in compress_layer_ids_for_test: # 只有被压缩的层长度会变
                     assert comp_cache_instance.get_seq_length(layer_idx) == expected_len_after_compress, \
                         f"L{layer_idx}: Expected len {expected_len_after_compress}, got {comp_cache_instance.get_seq_length(layer_idx)}"
                else: # 未参与压缩的层长度不变
                     assert comp_cache_instance.get_seq_length(layer_idx) == tokens_to_add_for_compression
    print(f"    _seen_tokens (should be unchanged by compression): {comp_cache_instance._seen_tokens}")

    print("\n--- Simple Test for CompCache (Global Multi-Layer Compression) Finished ---")