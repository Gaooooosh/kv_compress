import os
import torch
import torch.nn as nn
from .untis import CompressorConfig

class LearnedPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super(LearnedPoolingLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        weights = self.softmax(self.weights)
        pooled = torch.einsum('bid,d->bd', x, weights)  # 'bid': (batch, sequence_len, dim), 'd' adjusts with dim.
        return pooled  # Expected: torch.Size([32, 4096])

class KVCompressor(nn.Module):
    def __init__(self, config: CompressorConfig):
        super(KVCompressor, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=config.input_dim, out_channels=config.input_dim, 
                                kernel_size=3, stride=config.reduction_factor, padding=1)
        self.pool = LearnedPoolingLayer(config.input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=config.input_dim, num_heads=config.num_attention_heads, batch_first=True)
        self.fc = nn.Linear(config.input_dim, config.input_dim * config.output_seq_len)
        self.output_seq_len = config.output_seq_len
        self.use_mixed_precision = config.use_mixed_precision

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        batch_size, l, seq_len, dim = x.size()
        device = x.device
        autocast_device = device.type if self.use_mixed_precision else 'cpu'
        with torch.amp.autocast(device_type=autocast_device):
            x = x.view(batch_size * l, seq_len, dim).permute(0, 2, 1).to(device) # (32,4096,128)
            x = self.conv1d(x) # 32 4096 8
            x = x.permute(0, 2, 1) # 32 8 4096
            x = self.pool(x) # 32 4096 
            x = x.unsqueeze(1) # 32 1 4096
            attn_output, _ = self.attention(x, x, x) 

            x = self.fc(attn_output)
            x = x.view(batch_size, l, self.output_seq_len, dim)

        return x
if __name__ == '__main__':
    # Example usage with configuration
    config = CompressorConfig(
        input_dim=4096,
        reduction_factor=16,
        output_seq_len=2,
        num_attention_heads=4,
        use_mixed_precision=True
    )

    compressor = KVCompressor(config=config)

    # Device configuration (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compressor.to(device)

    # Generate dummy input data
    dummy_input = torch.randn((1, 32, 128, config.input_dim), device=device)
    output = compressor(dummy_input) # input shape (1, 32, 128, 4096) #(bcs, layer_num, seq_len, dim)
    print(output.shape)  # Expected shape: (1, 32, 2, 4096)