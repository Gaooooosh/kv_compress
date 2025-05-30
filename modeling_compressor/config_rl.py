# config_rl.py
# 该文件用于存储（现在主要是）压缩器训练、环境及LLM相关的配置参数

import torch

# 主要用于环境、LLM 和 KVCompressor 训练的参数
TRAINING_PARAMS = {
    # 环境及LLM相关
    "IDEAL_TOTAL_CONTEXT_LENGTH": 2048,  # LLM处理的最大上下文长度参考值
    "MIN_UNCOMPRESSED_CONTEXT_LENGTH": 1980, # 奖励函数中可能用到的参考值（当前极简版奖励简单）
    
    # 模拟LLM新token流入的参数 (在Environment中使用)
    "TOKENS_TO_GENERATE_PER_STEP": 1, # 环境每一步让LLM生成多少个新token
                                         # 为了精确对比logits，通常设为1
    "USE_LR_SCHEDULER":True,
    "LR_SCHEDULER_STEP_SIZE":5000.,
    "LR_SCHEDULER_GAMMA":0.5,
    # KVCompressor 训练相关参数
    "COMPRESSOR_LEARNING_RATE": 0.0002,   # KVCompressor的学习率
    "LOSS_FUNCTION": "KL",              # "KL" for KL Divergence, "MSE" for Mean Squared Error for logits comparison
    "KL_TEMPERATURE": 1.0,              # KL散度中softmax的温度参数（如果需要调整）
    "ACCUMULATION_STEPS" : 10,
    "EMA_ALPHA_LOGITS_REF": 0.1,
    # 训练过程相关参数
    "NUM_TRAINING_STEPS": 50000,          # 总训练步数 (替代 NUM_EPISODES)
    "MAX_TOKENS_PER_EPISODE": 128,       # 每个“回合”或数据段处理的最大token数量 (用于reset环境)
    "COMPRESSOR_MODEL_SAVE_FREQ_STEPS": 10000, # 每多少步保存一次压缩器模型
    "LOG_FREQ_STEPS": 100,                 # 每多少步打印一次日志
    "GRADIENT_CLIP_NORM": 0.0,            # 梯度裁剪的范数 (0表示不裁剪)
    # "LOAD_PRETRAINED_COMPRESSOR_PATH":"/root/autodl-tmp/compressor_training_output_dataset/saved_compressor_models/kv_compressor_step_40000.pth"
}

DATASET_CONFIG = {
    "dataset_name": "wikitext",
    "dataset_config_name": "wikitext-103-raw-v1", # 数据集子配置, e.g., "wikitext-103-raw-v1", "en" for c4
    "text_column": "text",              # 数据集中包含文本的列名
    "split": "train",                   # 使用哪个数据分割, e.g., "train", "validation"
    "max_samples_to_load": 10000,       # 加载的最大样本数量，用于限制数据集大小
    "min_text_length_for_sample": 256,  # 采样文本的最小长度 (tokenized)
    "max_text_length_for_sample": 1024, # 采样文本的最大长度 (tokenized) - 用于分段处理长文本
    "max_token_ized_length": TRAINING_PARAMS.get("IDEAL_TOTAL_CONTEXT_LENGTH", 512), # 使用理想上下文长度作为分块大小
    "stride_for_chunking": TRAINING_PARAMS.get("IDEAL_TOTAL_CONTEXT_LENGTH", 512) // 2, # 例如一半的重叠
    "prompt_min_len_ratio": 0.25,
    "prompt_max_len_ratio": 0.5,
}

TRAINING_PARAMS["dataset_args"] = DATASET_CONFIG # 将数据集配置嵌套进去


CURRICULUM_LEARNING_CONFIG = {
    "enabled": True,  # 是否启用课程学习
    "total_stages": 2, # 总共的训练阶段数

    "stage_1": {
        "enabled": True, # 是否启用第一阶段
        "duration_steps": 10000,  # 第一阶段持续的训练步数
        "dataset_args_override": { # 覆盖 DATASET_CONFIG 的参数
            "source_type": "huggingface_dataset",
            "hf_dataset_name": "wikitext",
            "hf_dataset_config_name": "wikitext-103-raw-v1", # 使用小数据集
            "hf_split": "train[:1%]", # 只用训练集的前10%
            "hf_text_column": "text",
            "max_samples_to_load": 200,  # 大幅减少样本量，增加重复性
            "min_raw_text_length": 200,    # 筛选长度适中的文本
            "max_raw_text_length": 256,    # 原始文本的最大字符长度，用于筛选更同质化的短文本
            "min_tokenized_length": 128,
            "max_tokenized_length": 256,   # 使用较短且固定的分块长度
            "stride_for_chunking": 128,     # 较大的重叠，增加数据片段间的相似性
            "prompt_min_len_ratio": 0.4,   # prompt比例可以稍大，让上下文更固定
            "prompt_max_len_ratio": 0.6,
        },
        "training_params_override": { # 覆盖 TRAINING_PARAMS 的参数
            "COMPRESSOR_LEARNING_RATE": 0.0001, # 第一阶段可以使用稍高或不同的学习率
            "KL_TEMPERATURE": 1.5,           # 可以用稍高的温度使目标分布更平滑
            "MAX_TOKENS_PER_EPISODE": 128,    # 每个文本块续写的token数可以少一些
        }
    },

    "stage_2": {
        # ... (第二阶段配置，可以使用 wikitext-103-raw-v1 的更大数据集和不同参数) ...
        "enabled": True,
        "duration_steps": 40000, 
        "dataset_args_override": {
            "source_type": "huggingface_dataset",
            "hf_dataset_name": "wikitext",
            "hf_dataset_config_name": "wikitext-103-raw-v1", # 第二阶段可以用更多数据
            "hf_split": "train[1%:50%]", # 例如使用1%到50%的数据
            "hf_text_column": "text",
            "max_samples_to_load": 10000, 
            "min_raw_text_length": 100,
            "min_tokenized_length": 128,
            "max_tokenized_length": TRAINING_PARAMS.get("IDEAL_TOTAL_CONTEXT_LENGTH", 512),
            "stride_for_chunking": TRAINING_PARAMS.get("IDEAL_TOTAL_CONTEXT_LENGTH", 512) // 2,
            "prompt_min_len_ratio": 0.25,
            "prompt_max_len_ratio": 0.5,
        },
        "training_params_override": {
            "COMPRESSOR_LEARNING_RATE": 0.00005, 
            "KL_TEMPERATURE": 1.0,
            "MAX_TOKENS_PER_EPISODE": TRAINING_PARAMS.get("MAX_TOKENS_PER_EPISODE", 256),
        }
    }
}
TRAINING_PARAMS["curriculum_learning_config"] = CURRICULUM_LEARNING_CONFIG


# LLM 相关配置 (保持不变)
LLM_CONFIG = {
    "model_name_or_path": "/root/autodl-tmp/tinyllama",
    "tokenizer_name_or_path": "/root/autodl-tmp/tinyllama",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # "device": "cpu",
    "max_new_tokens_per_llm_step": TRAINING_PARAMS["TOKENS_TO_GENERATE_PER_STEP"], # 与上面同步
    "max_context_length_for_llm_input": TRAINING_PARAMS["IDEAL_TOTAL_CONTEXT_LENGTH"],
}

# CompressorConfig 的默认参数 (可以由 untis.py 提供，这里是参考)
# 在创建CompressorConfig实例时，会从untis.py获取，RLEnvironment会尝试根据LLM覆盖部分
DEFAULT_COMPRESSOR_CONFIG_PARAMS = {
    "reduction_factor": 4,
    "output_seq_len": 4,
    "num_attention_heads": 8,
    "use_mixed_precision": False,
    "torch_dtype": "torch.float32", # 将在untis.py中转换为torch.dtype对象
    "kernel_size": 3,
    "padding": 1,
    "compression_threshold": 64, # CompCache中判断是否压缩的阈值 (固定规则)
    "initial_uncompressed_keep_length": 8
}
# 如果TRAINING_PARAMS中没有dataset_args，则使用此默认值
if "dataset_args" not in TRAINING_PARAMS:
    TRAINING_PARAMS["dataset_args"] = DATASET_CONFIG

if __name__ == '__main__':
    print("Training Parameters:", TRAINING_PARAMS)
    print("\nLLM Configuration:", LLM_CONFIG)
    print(f"\nLLM will run on: {LLM_CONFIG['device']}")
    print(f"Loss function for compressor training: {TRAINING_PARAMS['LOSS_FUNCTION']}")