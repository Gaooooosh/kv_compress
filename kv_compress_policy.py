import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Assuming CompressorConfig and your original KVCompressor's building blocks
# (LearnedPoolingLayer etc.) are accessible, e.g., from .model
from .model import CompressorConfig, LearnedPoolingLayer # Adjust import as needed

class KVCompressorPolicy(nn.Module):
    """
    The policy network (actor) for RL.
    Its core architecture is based on KVCompressor, but outputs distribution parameters.
    """
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.config = config
        self.param_dtype = config.torch_dtype

        # --- Core KVCompressor architecture ---
        self.conv1d = nn.Conv1d(
            in_channels=config.input_dim, out_channels=config.input_dim,
            kernel_size=getattr(config, 'kernel_size', 3),
            stride=config.reduction_factor, padding=getattr(config, 'padding', 1),
            dtype=self.param_dtype
        )
        self.pool = LearnedPoolingLayer(
            # This needs careful thought based on KVCompressor's actual pooling dimension
            # If pooling features, config.input_dim. If pooling sequence, it's the reduced seq_len.
            # For simplicity, let's assume it's designed to handle the output of conv1d correctly.
            # Let's assume your original KVCompressor's pool was designed to output (B*L, D_feat) eventually.
            # The input to pool layer is (B*L, S_reduced, D_feat).
            # If pool makes it (B*L, D_feat), then input_dim for attention is D_feat.
            # For now, we'll keep the structure abstractly.
            input_dim=config.input_dim, # Placeholder, adjust if pool changes feature dim
            layer_dtype=self.param_dtype
        )
        # This attention might be on the pooled sequence if pool outputs (B*L, 1, D_feat)
        # Or on the features themselves if pool reduces sequence to 1: (B*L, D_feat) -> unsqueeze -> (B*L, 1, D_feat)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.input_dim, num_heads=config.num_attention_heads,
            batch_first=True, dtype=self.param_dtype
        )
        # Base FC layer before policy/value heads
        # The output dimension here is arbitrary, choose something reasonable.
        self.fc_base_output_dim = config.input_dim # Example
        self.fc_base = nn.Linear(
            config.input_dim, # Input from attention
            self.fc_base_output_dim,
            dtype=self.param_dtype
        )
        # --- End Core KVCompressor architecture ---

        # Policy heads (Actor)
        # Output is compressed_dim = config.input_dim (features) * config.output_seq_len (new sequence length)
        self.compressed_feature_dim = config.input_dim * config.output_seq_len
        self.mu_head = nn.Linear(self.fc_base_output_dim, self.compressed_feature_dim, dtype=self.param_dtype)
        self.log_std_head = nn.Linear(self.fc_base_output_dim, self.compressed_feature_dim, dtype=self.param_dtype)

    def _forward_base(self, x_segments_for_compressor: torch.Tensor):
        # x_segments_for_compressor shape: (B_glob, L_glob, S_segment, D_orig_feat)
        # This is the output of CacheProcessor.prepare_segments_for_compression
        # KVCompressor's original forward expects (N, C_in, L_in) for conv1d
        # N = B_glob * L_glob, C_in = D_orig_feat, L_in = S_segment

        b_glob, l_glob, s_segment, d_orig_feat = x_segments_for_compressor.shape
        
        # Reshape for KVCompressor's internal processing path
        x = x_segments_for_compressor.view(b_glob * l_glob, s_segment, d_orig_feat)
        x = x.permute(0, 2, 1) # (N, D_orig_feat, S_segment) for conv1d

        x = self.conv1d(x)      # (N, D_orig_feat, S_compressed_conv)
        x = x.permute(0, 2, 1)  # (N, S_compressed_conv, D_orig_feat)
        
        # Pooling: This part needs to be consistent with how KVCompressor was designed
        # Assuming pool reduces S_compressed_conv or prepares for attention
        # If pool outputs (N, D_orig_feat) (e.g., global pooling over S_compressed_conv)
        # For this example, let's assume self.pool correctly processes x
        # and outputs something like (N, D_orig_feat) for simplicity
        # This means LearnedPoolingLayer's 'input_dim' should be S_compressed_conv, and einsum 'bsd,s->bd'
        # This is a BIG assumption and needs to match your actual KVCompressor design.
        # If pool keeps sequence dim: x = self.pool(x) -> (N, S_pooled, D_orig_feat)
        # Then x needs to be flattened or further processed before fc_base
        # For now, assume pooling + attention results in (N, config.input_dim) suitable for fc_base
        
        # Simplified path:
        # For this simplified example, let's assume pooling results in a fixed feature vector
        # This often means reducing the sequence dimension to 1 or a fixed small number.
        # A common pattern is GlobalAveragePooling1D after conv if sequence length is variable.
        # x = torch.mean(x, dim=1) # Global Average Pooling example, output (N, D_orig_feat)
        
        # If your LearnedPoolingLayer expects (N, S, D) and outputs (N, D_out_pool)
        x_pooled = self.pool(x) # Shape depends on your pool layer.
                                # For example, if it does 'bid,d->bd', output is (N, S_compressed_conv)
                                # This would be a mismatch for attention embed_dim=config.input_dim.
                                # This part MUST align with your actual KVCompressor structure.
                                # Let's assume pool is identity for now if it's too complex:
        # x_pooled = x.mean(dim=1) # (N, D_orig_feat) - Placeholder for pooling logic
        # Let's assume after permute, x is (N, S_comp_conv, D_feat), and we need to make it (N, D_feat)
        # for simpler attention/fc.
        if x_pooled.ndim == 3 and x_pooled.shape[1] != 1 : # if sequence still exists
            x_pooled = torch.mean(x_pooled, dim=1) # Global average pool over sequence
        elif x_pooled.ndim == 3 and x_pooled.shape[1] == 1:
             x_pooled = x_pooled.squeeze(1)


        # Attention needs (N, L_att, E_att). If x_pooled is (N, D_orig_feat), L_att=1.
        att_input = x_pooled.unsqueeze(1) # (N, 1, D_orig_feat)
        attn_output, _ = self.attention(att_input, att_input, att_input) # (N, 1, D_orig_feat)
        fc_input = attn_output.squeeze(1) # (N, D_orig_feat)
        
        base_features = F.relu(self.fc_base(fc_input)) # (N, self.fc_base_output_dim)
        
        # Reshape base_features back to (b_glob, l_glob, self.fc_base_output_dim)
        base_features = base_features.view(b_glob, l_glob, -1)
        return base_features

    def forward(self, x_segments_for_compressor: torch.Tensor, sample_action: bool = True):
        """
        Args:
            x_segments_for_compressor (torch.Tensor): Batch of uncompressed segments.
                Shape: (batch_size_llm, num_segments_in_batch, segment_seq_len, kv_compressor_input_dim)
            sample_action (bool): If True, sample from distribution. Else, take mean.
        Returns:
            action_samples (torch.Tensor): Compressed segments.
                Shape: (batch_size_llm, num_segments_in_batch, self.compressed_feature_dim)
            log_probs (torch.Tensor): Log probability of the sampled actions. Shape: (batch_size_llm, num_segments_in_batch)
            entropy (torch.Tensor): Entropy of the distribution. Shape: (batch_size_llm, num_segments_in_batch)
        """
        # Pass through the base network (shared part of original KVCompressor)
        # The input x_segments_for_compressor is what CacheProcessor.prepare_segments_for_compression outputs
        base_features = self._forward_base(x_segments_for_compressor)
        # base_features shape: (B_glob, L_glob, self.fc_base_output_dim)

        mu = self.mu_head(base_features) # (B_glob, L_glob, self.compressed_feature_dim)
        log_std = self.log_std_head(base_features) # (B_glob, L_glob, self.compressed_feature_dim)
        std = torch.exp(log_std.clamp(-20, 2)) # Clamp for stability

        dist = Normal(mu, std)

        if sample_action:
            # Reparameterization trick: sampled_action = mu + std * N(0,1)
            action_samples = dist.rsample()
        else:
            action_samples = mu # For evaluation or deterministic action

        log_probs = dist.log_prob(action_samples).sum(dim=-1) # Sum over the feature dimension
        entropy = dist.entropy().sum(dim=-1) # Sum over the feature dimension

        # action_samples now has shape (B_glob, L_glob, self.compressed_feature_dim)
        # This needs to be reshaped by CacheProcessor.format_compressed_segments_output
        # back into list of (B_glob, model_num_kv_heads, compressed_seq_len, model_head_dim_per_kv_head)
        # This means self.compressed_feature_dim must be consistent with that.
        # compressed_feature_dim = model_num_kv_heads * config.output_seq_len * model_head_dim_per_kv_head
        # No, this is simpler: self.compressed_feature_dim = config.input_dim * config.output_seq_len (total features)
        # KVCompressor output was (B, L, S_out, D_feat)
        # Here, action_samples is (B, L_glob, S_out*D_feat if flattened).
        # The KVCompressor's output was (batch_size_kv, layers_kv, self.output_seq_len, dim_features_kv)
        # So, our self.compressed_feature_dim must be output_seq_len * dim_features_kv.
        # This implies action_samples is essentially the flattened version of one compressed segment's features over its new sequence.

        # Let's assume `action_samples` is effectively (batch_size_llm, num_segments_in_batch, config.output_seq_len, config.input_dim)
        # for consistency with KVCompressor output that `format_compressed_segments_output` expects.
        # So, self.compressed_feature_dim = config.output_seq_len * config.input_dim
        # The heads should output this total dimension.
        # Then we reshape it before returning or CacheProcessor does it.
        
        # If action_samples is (B_glob, L_glob, S_out_compressed * D_feat_compressed_total),
        # it means it's already in the "glob" format that KVCompressor used to output before final view.
        # The KVCompressor's final output (before to_list) was (B_glob, L_glob, S_out_comp, D_feat_comp)
        # where L_glob is num_segments here.
        # So, mu and log_std should output features for S_out_comp * D_feat_comp

        # The CacheProcessor.format_compressed_segments_output expects input:
        # (batch_size, num_segments_processed, compressed_segment_seq_len, config.input_dim)
        # So our actor should output this shape for action_samples.
        # This implies self.compressed_feature_dim should be structured.
        # mu_head outputs (..., config.output_seq_len * config.input_dim)
        # Then reshape mu and log_std to (..., config.output_seq_len, config.input_dim)
        
        action_samples = action_samples.view(
            *mu.shape[:-1], # (B_glob, L_glob)
            self.config.output_seq_len, 
            self.config.input_dim
        )

        return action_samples, log_probs, entropy


class ValueNetwork(nn.Module):
    """
    The Critic network for PPO.
    Estimates the value of a state (uncompressed segment).
    """
    def __init__(self, config: CompressorConfig):
        super().__init__()
        self.config = config
        self.param_dtype = config.torch_dtype

        # Reuse the base architecture from KVCompressorPolicy
        # Or define a simpler MLP if state is pre-processed
        # For simplicity, let's assume it can take the same raw segment batch
        # and use a similar base.
        self.policy_base = KVCompressorPolicy(config) # Get the _forward_base
        # Detach policy_base from ValueNetwork's computation graph for parameters if sharing,
        # or create a separate instance. For simplicity: separate instance for now.
        self.base_processor = KVCompressorPolicy(config)._forward_base 
        
        # Value head
        self.value_head = nn.Linear(
            self.policy_base.fc_base_output_dim, # fc_base_output_dim from KVCompressorPolicy
            1, # Outputs a single scalar value
            dtype=self.param_dtype
        )

    def forward(self, x_segments_for_compressor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_segments_for_compressor (torch.Tensor): Batch of uncompressed segments.
                Shape: (batch_size_llm, num_segments_in_batch, segment_seq_len, kv_compressor_input_dim)
        Returns:
            torch.Tensor: Estimated value of the state. Shape: (batch_size_llm, num_segments_in_batch, 1)
        """
        base_features = self.base_processor(x_segments_for_compressor)
        # base_features shape: (B_glob, L_glob, self.policy_base.fc_base_output_dim)
        
        value = self.value_head(base_features) # (B_glob, L_glob, 1)
        return value