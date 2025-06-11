# untis.py
from dataclasses import dataclass
from typing import Any, List, Optional
import torch # 确保导入 torch 以便 CompressorConfig 中的 torch_dtype 能被正确解析

@dataclass
class CompressorConfig:
    # 使用您上传的版本，确保 __init__ 中的类型提示和默认值是合理的
    def __init__(self, input_dim, reduction_factor, output_seq_len,
                 num_attention_heads, use_mixed_precision, torch_dtype, # num_attention_heads 是KVCompressor内部MHA的头数(如果使用)
                 kv_head_dim, layer_nums, # kv_head_dim 是LLM的KV头数, layer_nums 是LLM的总层数
                 kernel_size=3, padding=1,
                 compression_threshold: int = 16,
                 compress_layer_ids: Optional[List[int]] = None,
                 initial_uncompressed_keep_length: int = 0
                ):
        self.input_dim = input_dim # 特征维度 D_feat = head_dim * kv_head_dim (LLM的)
        self.reduction_factor = reduction_factor # Conv1d的stride，用于序列长度缩减
        self.output_seq_len = output_seq_len # AdaptivePool1d的目标输出序列长度
        self.num_attention_heads = num_attention_heads # KVCompressor内部注意力机制的头数 (如果模型中有)
        self.use_mixed_precision = use_mixed_precision
        
        # 处理 torch_dtype 字符串到 torch.dtype 对象
        if isinstance(torch_dtype, str):
            dtype_map = {
                "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
                "torch.float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16, "float32": torch.float32
            }
            resolved_dtype = dtype_map.get(torch_dtype.lower())
            if resolved_dtype is None:
                print(f"Warning (CompressorConfig): Unrecognized torch_dtype string '{torch_dtype}'. Defaulting to torch.float32.")
                self.torch_dtype = torch.float32
            else:
                self.torch_dtype = resolved_dtype
        elif isinstance(torch_dtype, torch.dtype):
            self.torch_dtype = torch_dtype
        else:
            print(f"Warning (CompressorConfig): Invalid torch_dtype type '{type(torch_dtype)}'. Defaulting to torch.float32.")
            self.torch_dtype = torch.float32
        
        self.kv_head_dim = kv_head_dim # LLM的每个KV头的维度 (注意：这里命名可能混淆，通常kv_head_dim指维度，num_kv_heads指导数量)
                                      # 假设这里指的是LLM的KV头数量 (num_kv_heads)
        self.layer_nums = layer_nums # LLM的总层数，用于 CompCache 初始化列表长度
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.initial_uncompressed_keep_length = initial_uncompressed_keep_length
        self.compression_threshold = compression_threshold # 每层提取用于压缩的段的长度

        if compress_layer_ids is None and layer_nums is not None:
            self.compress_layer_ids = list(range(layer_nums))
        elif compress_layer_ids is not None:
            self.compress_layer_ids = [lid for lid in compress_layer_ids if 0 <= lid < (layer_nums if layer_nums is not None else float('inf'))]
        else: # layer_nums is None and compress_layer_ids is None
            self.compress_layer_ids = [] # 或者抛出错误，因为不知道哪些层可以压缩

        # print(f"CompressorConfig initialized with torch_dtype: {self.torch_dtype}")