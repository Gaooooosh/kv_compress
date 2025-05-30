import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming CompressorConfig is in .untis, if not, adjust import
# from .untis import CompressorConfig 

# Placeholder for CompressorConfig if not imported, ensure it matches your definition
class CompressorConfig:
    def __init__(self, input_dim, reduction_factor, output_seq_len,
                 num_attention_heads, use_mixed_precision, torch_dtype,
                 kv_head_dim=None, layer_nums=None, compress_layers=None, # Added from your infra.py
                 kernel_size=3, padding=1): # Sensible defaults
        self.input_dim = input_dim
        self.reduction_factor = reduction_factor
        self.output_seq_len = output_seq_len
        self.num_attention_heads = num_attention_heads
        self.use_mixed_precision = use_mixed_precision
        self.torch_dtype = torch_dtype # This is key
        self.kv_head_dim = kv_head_dim
        self.layer_nums = layer_nums
        self.compress_layers = compress_layers
        self.kernel_size = kernel_size
        self.padding = padding


class LearnedPoolingLayer(nn.Module):
    def __init__(self, input_dim, layer_dtype=torch.float32):
        """
        Args:
            input_dim (int): The size of the dimension to apply learned weights for pooling.
                             Based on original einsum 'bid,d->bd', this is the feature dimension.
            layer_dtype (torch.dtype): The data type for the layer's parameters.
        """
        super(LearnedPoolingLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim, dtype=layer_dtype))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weights match the (potentially autocast-ed) dtype of x
        # Softmax preserves dtype for float inputs.
        weights = self.softmax(self.weights.to(dtype=x.dtype))
        
        # Original einsum: 'bid,d->bd'
        # x shape: (batch, sequence_len, dim_features). self.weights shape: (dim_features)
        # Output shape: (batch, sequence_len). This pools/reduces the feature dimension.
        # This contradicts the comment # Expected: torch.Size([32, 4096]) if 4096 is feature dim
        # and input is (32, 8, 4096). This einsum would give (32, 8).
        # If the goal is (batch, features_dim), the einsum or layer logic needs to change
        # to pool over the sequence dimension. E.g., 'bsd,s->bd' if weights are (sequence_len).
        # For now, fixing dtype error while keeping original einsum.
        pooled = torch.einsum('bid,d->bd', x, weights)
        return pooled


class KVCompressor(nn.Module):
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.config = config
        D_feat = config.input_dim
        D_feat_intermediate_conv = D_feat * 2 
        reduction_factor_conv_stride = config.reduction_factor # 假设主要长度缩减在conv2
        self.output_seq_len_pool = config.output_seq_len
        
        # --- 卷积层 ---
        self.conv1 = nn.Conv1d(D_feat, D_feat_intermediate_conv, kernel_size=3, stride=1, padding=1)
        self.norm_conv1 = nn.LayerNorm(D_feat_intermediate_conv) # 作用于特征维
        
        # Conv2 - stride会缩短序列长度
        # 计算conv2的padding以尽量维持长度，或者让AdaptiveAvgPool1d处理
        # 如果stride=2, kernel=3, padding=1, L_out = floor((L_in + 2*1 - 1*(3-1) -1)/2 + 1) = floor(L_in/2)
        self.conv2 = nn.Conv1d(D_feat_intermediate_conv, D_feat, kernel_size=3, 
                               stride=reduction_factor_conv_stride, 
                               padding=1) 
        self.norm_conv2 = nn.LayerNorm(D_feat)

        # --- 池化层 ---
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.output_seq_len_pool)

        # --- MLP层 ---
        D_mlp_hidden = D_feat * 2
        self.fc1 = nn.Linear(D_feat, D_mlp_hidden)
        self.mlp_activation = nn.GELU()
        self.fc2 = nn.Linear(D_mlp_hidden, D_feat)
        self.norm_mlp_out = nn.LayerNorm(D_feat) # 对MLP的输出进行归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, Num_P_Layers, S_orig, D_feat)
        B, Num_P_Layers, S_orig, D_feat_in = x.shape
        
        # Reshape for Conv1d: (N, D_feat_in, S_orig) where N = B*Num_P_Layers
        x_for_conv = x.view(-1, S_orig, D_feat_in).permute(0, 2, 1) 
        N = x_for_conv.shape[0]

        # --- 卷积模块 ---
        # Layer 1
        c1_out = self.conv1(x_for_conv) # (N, D_feat_intermediate, S_orig)
        c1_norm = self.norm_conv1(c1_out.permute(0, 2, 1)).permute(0, 2, 1) # Norm on features
        c1_act = F.gelu(c1_norm)
        # 可以加入残差: c1_act = c1_act + self.conv1_shortcut(x_for_conv) 

        # Layer 2
        c2_out = self.conv2(c1_act) # (N, D_feat, S_reduced_by_conv)
        c2_norm = self.norm_conv2(c2_out.permute(0, 2, 1)).permute(0, 2, 1) # Norm on features
        conv_final_out = F.gelu(c2_norm)
        # 可以加入残差: conv_final_out = conv_final_out + self.conv2_shortcut(c1_act_ appropriately_sized)
        
        # --- 池化模块 ---
        # adaptive_pool expects (N, C, L_in) -> (N, C, L_out)
        pooled_features = self.adaptive_pool(conv_final_out) # (N, D_feat, output_seq_len)
        
        # Transpose for MLP: (N, output_seq_len, D_feat)
        sequence_for_mlp = pooled_features.permute(0, 2, 1) 

        # --- MLP模块 ---
        identity_mlp = sequence_for_mlp
        mlp_out1 = self.fc1(sequence_for_mlp)
        mlp_out1_act = self.mlp_activation(mlp_out1)
        
        mlp_final_features = self.fc2(mlp_out1_act) # (N, output_seq_len, D_feat)
        mlp_final_res = mlp_final_features + identity_mlp # 残差连接
        mlp_output_norm = self.norm_mlp_out(mlp_final_res) # 输出归一化
        
        # Reshape back to (B, Num_P_Layers, output_seq_len, D_feat)
        final_output = mlp_output_norm.view(B, Num_P_Layers, self.output_seq_len_pool, D_feat_in)
        
        return final_output

if __name__ == '__main__':
    # Example usage with configuration from your infra.py
    # This CompressorConfig matches the one in your infra.py for testing
    cfg = CompressorConfig(
        input_dim=4096, # Example, will be head_dim * kv_head_num
        reduction_factor=4,
        output_seq_len=8,
        num_attention_heads=8, # Example, will be kv_head_num
        use_mixed_precision=True,
        torch_dtype=torch.bfloat16, # As per your infra.py
        kv_head_dim=32, # Example
        layer_nums=32, # Example
        kernel_size=3,
        padding=1
    )
    # Adjust input_dim and num_attention_heads based on typical Llama setup
    # For Llama-3-8B, n_heads = 32, n_kv_heads = 8, head_dim = 128
    # So, if input_dim is for one head's features before combining, it's head_dim.
    # If it's after combining KV heads for compression, it's n_kv_heads * head_dim.
    # Your infra.py uses: input_dim=head_dim*kv_head_num
    # num_attention_heads=kv_head_num (for the KVCompressor's internal attention)

    # Let's use values that might align with Llama-3-8B structure for KV cache processing
    # Say, we are processing KV cache where each of the 8 KV heads has dim 128.
    # config.input_dim = 8 * 128 = 1024 if processing combined KV heads for a layer.
    # Or if config.input_dim is the main model's hidden_dim (4096 for Llama-3-8B)
    # and the KVCompressor operates on this full dimension (unlikely for per-head KV cache).
    # Your infra.py:
    # _, kv_head_num, seq_len, head_dim = cache[0][0].size() # (batch, num_kv_heads, seq, head_dim)
    # compressor_config.input_dim = head_dim * kv_head_num
    # compressor_config.num_attention_heads = kv_head_num (for KVCompressor's own attention)

    # Let's simulate this:
    actual_head_dim = 128
    actual_kv_head_num = 8
    
    test_config = CompressorConfig(
        input_dim=actual_head_dim * actual_kv_head_num, # 1024
        reduction_factor=4, 
        output_seq_len=8, 
        num_attention_heads=actual_kv_head_num, # KVCompressor's internal MHA uses 8 heads
        use_mixed_precision=True,
        torch_dtype=torch.bfloat16, # Your target
        kv_head_dim=actual_kv_head_num, # This seems to be a misnomer in your config, should be num_kv_heads
        layer_nums = 32, # Llama-3-8B has 32 layers
        compress_layers = [],
        kernel_size=3,
        padding=1
    )

    compressor = KVCompressor(config=test_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' and test_config.torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Warning: bfloat16 requested but not supported. Switching parameters to float32.")
        test_config.torch_dtype = torch.float32
        # Re-init with float32 if bfloat16 not supported and it's critical for params
        compressor = KVCompressor(config=test_config) 
        # Autocast will still try bfloat16 if enabled and device supports it for ops,
        # or fallback. Or, you might disable mixed precision.
        # For this test, let's assume parameters are now float32.
        # test_config.use_mixed_precision = False # Option if bfloat16 is problematic

    compressor.to(device)
    # Crucially, cast the model parameters to the target dtype
    compressor.to(dtype=test_config.torch_dtype)
    print(f"Compressor initialized on {device} with parameter dtype {next(compressor.parameters()).dtype}")


    # Dummy input based on CacheProcessor output:
    # processed_cache = combined_cache.view(batch_size, layers_num, seq_len, -1)
    # -1 is head_dim * kv_head_num (config.input_dim for compressor)
    # So input to KVCompressor is (batch_size, layers_num, seq_len_orig_cache, config.input_dim)
    
    # Example from your infra.py:
    # dummy_input = torch.randn((1, 32, 128, config.input_dim), device=device)
    # This matches (batch_size, l_layers, seq_len_orig, dim_features)
    
    example_batch_size = 1
    example_l_layers = 32 # num_layers from Llama model
    example_seq_len_orig = 60 # Original sequence length of a cache segment
    
    dummy_input_tensor = torch.randn(
        (example_batch_size, example_l_layers, example_seq_len_orig, test_config.input_dim),
        device=device
    )
    # If not using mixed precision, input should match param_dtype
    if not test_config.use_mixed_precision:
        dummy_input_tensor = dummy_input_tensor.to(test_config.torch_dtype)
    
    print(f"Dummy input shape: {dummy_input_tensor.shape}, dtype: {dummy_input_tensor.dtype}")

    try:
        output = compressor(dummy_input_tensor)
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")
        # Expected shape: (batch_size, l_layers, output_seq_len, config.input_dim)
        # (1, 32, 8, 1024)
        assert output.shape == (example_batch_size, example_l_layers, test_config.output_seq_len, test_config.input_dim)
        print("Test forward pass successful.")
    except Exception as e:
        print(f"Error during test forward pass: {e}")
        import traceback
        traceback.print_exc()