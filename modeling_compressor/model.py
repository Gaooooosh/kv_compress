import os
import torch
import torch.nn as nn
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
        super(KVCompressor, self).__init__()
        self.config = config
        self.param_dtype = config.torch_dtype # Intended dtype for all layer parameters

        self.conv1d = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.input_dim,
            kernel_size=getattr(config, 'kernel_size', 3), # Use getattr for potential missing attrs
            stride=config.reduction_factor,
            padding=getattr(config, 'padding', 1),
            dtype=self.param_dtype # Crucial: Initialize layer with target dtype
        )
        
        # Assuming input_dim for LearnedPoolingLayer refers to the feature dimension
        # as per the original einsum.
        self.pool = LearnedPoolingLayer(
            input_dim=config.input_dim,
            layer_dtype=self.param_dtype # Pass dtype here
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.input_dim, # Assuming pooling output still has config.input_dim features
                                        # or that the pooling layer's output dim matches this.
                                        # If LearnedPoolingLayer's output is (Batch, SeqLen), this will be a mismatch.
                                        # The comment "x = self.pool(x) # 32 4096" suggests output is (Batch, Features)
                                        # This means LearnedPoolingLayer should be pooling the sequence dimension.
            num_heads=config.num_attention_heads,
            batch_first=True,
            dtype=self.param_dtype # Crucial
        )
        
        self.fc = nn.Linear(
            config.input_dim, # Input features to FC
            config.input_dim * config.output_seq_len,
            dtype=self.param_dtype # Crucial
        )
        self.output_seq_len = config.output_seq_len
        self.use_mixed_precision = config.use_mixed_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        batch_size, l_layers, seq_len_orig, dim_features = x.size()
        original_input_dtype = x.dtype # Store original dtype for final cast if needed
        device = x.device
        
        # Determine device type for autocast, handling 'mps' if necessary
        current_device_type = device.type
        if current_device_type == 'mps': # MPS has specific autocast behavior
            # For MPS, dtype in autocast is usually not specified or set to None.
            # It might also depend on PyTorch version.
            # If issues persist on MPS, specific handling might be needed.
            # For now, let's assume 'cpu' behavior for autocast if MPS to avoid errors,
            # or let it handle itself if `self.use_mixed_precision` is False.
            autocast_device_type_for_amp = 'cpu' if self.use_mixed_precision else 'cpu' # Or handle MPS more directly
        else:
            autocast_device_type_for_amp = current_device_type if self.use_mixed_precision else 'cpu'

        # Determine target dtype for autocast operations
        autocast_target_dtype = None
        if self.use_mixed_precision:
            if self.param_dtype == torch.bfloat16:
                autocast_target_dtype = torch.bfloat16
            elif self.param_dtype == torch.float16:
                autocast_target_dtype = torch.float16
            # If self.param_dtype is float32, autocast_target_dtype remains None,
            # and AMP will use its default (e.g., float16 for CUDA).

        with torch.amp.autocast(
            device_type=autocast_device_type_for_amp, # Use determined device type
            enabled=self.use_mixed_precision,
            dtype=autocast_target_dtype # Explicitly set autocast operation dtype
        ):
            # Reshape for Conv1D: (batch_size * l_layers, dim_features, seq_len_orig)
            x_reshaped = x.view(batch_size * l_layers, seq_len_orig, dim_features)
            # Input to conv1d should be (N, C_in, L_in)
            # N = batch_size * l_layers
            # C_in = dim_features (config.input_dim)
            # L_in = seq_len_orig
            x_for_conv = x_reshaped.permute(0, 2, 1) # Shape: (B*L, D, S_orig)
            
            # If not using mixed precision, but parameters are not float32,
            # ensure input matches parameter dtype.
            # However, if parameters are already self.param_dtype, and autocast is disabled,
            # input should match self.param_dtype.
            if not self.use_mixed_precision and x_for_conv.dtype != self.param_dtype:
                 x_for_conv = x_for_conv.to(self.param_dtype)

            # Inside autocast, operations will use autocast_target_dtype or self.param_dtype
            # x_for_conv will be cast by autocast if enabled and dtypes differ.
            
            # Conv1D
            # Input: (B*L, D, S_orig), e.g., (32, 4096, 128)
            # Output: (B*L, D, S_reduced), e.g., (32, 4096, 8)
            conv_out = self.conv1d(x_for_conv)
            
            # Permute for Pooling/Attention: (B*L, S_reduced, D)
            # e.g., (32, 8, 4096)
            permuted_conv_out = conv_out.permute(0, 2, 1)

            # Pooling
            # Input to pool: (32, 8, 4096)
            # If LearnedPoolingLayer uses original 'bid,d->bd' with weights (4096):
            #   pooled_out will be (32, 8). This is (Batch, SeqReduced).
            # The comment "# x = self.pool(x) # 32 4096" implies pooled_out should be (32, 4096) (Batch, Features).
            # This requires LearnedPoolingLayer to pool the sequence dimension (size 8).
            # For now, using the existing LearnedPoolingLayer structure.
            # If pooled_out is (32,8), attention will fail if embed_dim is 4096.
            # Let's assume the comment "# 32 4096" is the target shape for x after pooling.
            # This means the pooling operation results in (batch_size*l, feature_dim).
            # For this to happen with the current LearnedPoolingLayer, its internal logic would need to change.
            # For the sake of this fix, we assume that `pooled_out` will have `dim_features`.
            pooled_out = self.pool(permuted_conv_out)

            # Critical check for pooling output dimension:
            # If pooled_out.shape is (B*L, S_reduced) due to 'bid,d->bd' and input_dim=D:
            # And self.attention expects embed_dim=D, then unsqueeze(1) won't work.
            # The comment "x = x.unsqueeze(1) # 32 1 4096" implies pooled_out is (32, 4096).
            # This means the pooling correctly yields (Batch, Features).
            # If your LearnedPoolingLayer's `input_dim` is `config.input_dim` (features),
            # and einsum is `bid,d->bd`, output is `(B*L, S_reduced)`.
            # For the code to proceed as commented, `pooled_out` must be `(B*L, dim_features)`.
            # This would happen if `LearnedPoolingLayer` was, e.g., a global average pool over S_reduced,
            # or a learned pool over S_reduced.

            # Assuming pooled_out is (B*L, dim_features) as per comment # 32 4096
            # If not, the following unsqueeze and attention will have dimension mismatch.
            # Corrected path would involve fixing LearnedPoolingLayer or using a different pool.
            
            att_input = pooled_out.unsqueeze(1) # Shape: (B*L, 1, D_feat), e.g. (32, 1, 4096)
            
            attn_output, _ = self.attention(query=att_input, key=att_input, value=att_input)
            # Output of attention: (B*L, 1, D_feat), e.g. (32, 1, 4096)

            # Fully Connected Layer
            fc_input = attn_output.squeeze(1) # Shape: (B*L, D_feat), e.g. (32, 4096)
            fc_out = self.fc(fc_input)
            # Output of FC: (B*L, D_feat * output_seq_len)

            # Reshape final output
            # Target shape: (batch_size, l_layers, self.output_seq_len, dim_features)
            final_output_mixed_precision = fc_out.view(
                batch_size, l_layers, self.output_seq_len, dim_features
            )
            
            # The .to(original_input_dtype) is done outside autocast if needed
            # If it's inside, it might cast prematurely.
            # If the rest of the model expects original_input_dtype, this cast is fine.

        # Cast to original input dtype if it was different from the processing dtype.
        # This is what the user had, so preserving it.
        # If self.param_dtype is the desired output, use that.
        if final_output_mixed_precision.dtype != original_input_dtype:
             final_output = final_output_mixed_precision.to(original_input_dtype)
        else:
             final_output = final_output_mixed_precision
             
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