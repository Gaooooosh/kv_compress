# config_rl.py
# 该文件用于存储（现在主要是）压缩器训练、环境及LLM相关的配置参数

import torch

# 主要用于环境、LLM 和 KVCompressor 训练的参数
TRAINING_PARAMS = {
    # 环境及LLM相关
    "IDEAL_TOTAL_CONTEXT_LENGTH": 1024,  # LLM处理的最大上下文长度参考值
    "MIN_UNCOMPRESSED_CONTEXT_LENGTH": 256, # 奖励函数中可能用到的参考值（当前极简版奖励简单）
    
    # 模拟LLM新token流入的参数 (在Environment中使用)
    "TOKENS_TO_GENERATE_PER_STEP": 1, # 环境每一步让LLM生成多少个新token
                                         # 为了精确对比logits，通常设为1

    # KVCompressor 训练相关参数
    "COMPRESSOR_LEARNING_RATE": 0.0001,   # KVCompressor的学习率
    "LOSS_FUNCTION": "KL",              # "KL" for KL Divergence, "MSE" for Mean Squared Error for logits comparison
    "KL_TEMPERATURE": 1.0,              # KL散度中softmax的温度参数（如果需要调整）

    # 训练过程相关参数
    "NUM_TRAINING_STEPS": 50000,          # 总训练步数 (替代 NUM_EPISODES)
    "MAX_TOKENS_PER_EPISODE": 512,       # 每个“回合”或数据段处理的最大token数量 (用于reset环境)
    "COMPRESSOR_MODEL_SAVE_FREQ_STEPS": 1000, # 每多少步保存一次压缩器模型
    "LOG_FREQ_STEPS": 10,                 # 每多少步打印一次日志
    "GRADIENT_CLIP_NORM": 1.0,            # 梯度裁剪的范数 (0表示不裁剪)
}

# LLM 相关配置 (保持不变)
LLM_CONFIG = {
    "model_name_or_path": "/raid_sdh/home/xyg/PRETRAINED_MODEL/TinyLlama-chat",
    "tokenizer_name_or_path": "/raid_sdh/home/xyg/PRETRAINED_MODEL/TinyLlama-chat",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_new_tokens_per_llm_step": TRAINING_PARAMS["TOKENS_TO_GENERATE_PER_STEP"], # 与上面同步
    "max_context_length_for_llm_input": TRAINING_PARAMS["IDEAL_TOTAL_CONTEXT_LENGTH"],
}

# CompressorConfig 的默认参数 (可以由 untis.py 提供，这里是参考)
# 在创建CompressorConfig实例时，会从untis.py获取，RLEnvironment会尝试根据LLM覆盖部分
DEFAULT_COMPRESSOR_CONFIG_PARAMS = {
    "reduction_factor": 4,
    "output_seq_len": 2,
    "num_attention_heads": 8,
    "use_mixed_precision": False,
    "torch_dtype": "torch.float32", # 将在untis.py中转换为torch.dtype对象
    "kernel_size": 3,
    "padding": 1,
    "compression_threshold": 10, # CompCache中判断是否压缩的阈值 (固定规则)
    "initial_uncompressed_keep_length": 4
}


if __name__ == '__main__':
    print("Training Parameters:", TRAINING_PARAMS)
    print("\nLLM Configuration:", LLM_CONFIG)
    print(f"\nLLM will run on: {LLM_CONFIG['device']}")
    print(f"Loss function for compressor training: {TRAINING_PARAMS['LOSS_FUNCTION']}")