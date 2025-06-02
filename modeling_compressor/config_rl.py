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
    "LR_SCHEDULER_STEP_SIZE":5000,
    "LR_SCHEDULER_GAMMA":0.5,
    # KVCompressor 训练相关参数
    "COMPRESSOR_LEARNING_RATE": 0.0002,   # KVCompressor的学习率
    "LOSS_FUNCTION": "TOP_K_KL",  # "TOP_K_KL"、"TOP_K_MSE"
    "KL_TEMPERATURE": 1.0,              # KL散度中softmax的温度参数（如果需要调整）
    "TOP_K_LOGITS": 10,           # only for TOP_K_KL 选择 Top-K 个 logits 进行比较，例如 K=10
    "KL_TEMPERATURE": 1.0,        # 保持，因为Top-K KL仍然可以用到
    "ACCUMULATION_STEPS" : 10,
    "EMA_ALPHA_LOGITS_REF": 0.1,
    # 训练过程相关参数
    "NUM_TRAINING_STEPS": 50000,          # 总训练步数 (替代 NUM_EPISODES)
    "MAX_TOKENS_PER_EPISODE": 128,       # 每个“回合”或数据段处理的最大token数量 (用于reset环境)
    "COMPRESSOR_MODEL_SAVE_FREQ_STEPS": 10000, # 每多少步保存一次压缩器模型
    "LOG_FREQ_STEPS": 100,                 # 每多少步打印一次日志
    "GRADIENT_CLIP_NORM": 0.0,            # 梯度裁剪的范数 (0表示不裁剪)
    "ALTERNATING_TRAINING_MODE": "block", # "block" (每N步切换) 或 "step" (每步轮流)
    "ALTERNATING_BLOCK_SIZE": 5000,        # 如果 mode="block", 每多少步切换一次训练K还是V
    "MODEL_SAVE_DIR":"/raid_sdh/home/xyg/compressor_training_output_dataset"
    # "LOAD_PRETRAINED_COMPRESSOR_PATH":""
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
        "duration_steps": 20000,  # 第一阶段持续的训练步数
        "dataset_args_override": { # 覆盖 DATASET_CONFIG 的参数
            # "source_type": "huggingface_dataset",
            "source_type": "fixed_list",
            "fixed_texts":["Zeng Laishun was born around 1826 or 1827 in Singapore. His father was a Teochew migrant from eastern Guangdong, a province of southern China, and his mother was Malay. Zeng was brought up mainly speaking Malay. Both of his parents worked as vegetable farmers, and died when he was a young child. Orphaned, he was sent to work serving tables at the American Consulate. There, in 1836, he was noticed by American Board of Commissioners for Foreign Missions missionary Joseph Travelli, who had him enrolled in a Chinese day school established by his colleague Ira Tracy the previous year. The school's missionaries referred to him as \"Chan Laisun\".Around April 1843, Zeng was sent to the United States to continue his education. He was accompanied by the Presbyterian missionary John Hunter Morrison, who was returning to America after work in northern India. They went west via the Indian and Atlantic oceans, sailing around the Cape of Good Hope to dock on the east coast of the country. Morrison raised funds from among his friends and enrolled Zeng in the Bloomfield Academy, a boarding school in Bloomfield, New Jersey. Morrison returned to India in 1846, and Zeng was put into the care of an American Board missionary previously stationed in Guangzhou, Samuel Wells Williams. In the fall of 1846, Zeng transferred from Bloomfield to Hamilton College, a Presbyterian institution in Clinton, New York. Williams arranged for the First Presbyterian Church in Utica, New York, to support Zeng's study for two years, the \"faculty offering to teach him gratuitously, and the ladies in Brooklyn to clothe him.\".According to historian Edward J. M. Rhoads, Zeng was the first Chinese person to attend college in the United States, and possibly the first at any foreign college. At Hamilton, Zeng studied the New Testament in Koine Greek (likely under classicist Edward North) and taught Sunday school at a local church. He was active in the college's glee club. In early 1848, his funding ran out, and he was forced to withdraw from the college. Williams attempted to arrange for the American Board to take Zeng to China to work as a teacher at their mission in Xiamen, but the American Board refused, stating that it was the Presbyterian Church's responsibility to transport him, and that foreign-educated Chinese Christians were unsuitable for mission work. Instead, he traveled to China with Williams and his wife, departing from New York City in late May 1848.Zeng Laishun was born around 1826 or 1827 in Singapore. His father was a Teochew migrant from eastern Guangdong, a province of southern China, and his mother was Malay. Zeng was brought up mainly speaking Malay. Both of his parents worked as vegetable farmers, and died when he was a young child. Orphaned, he was sent to work serving tables at the American Consulate. There, in 1836, he was noticed by American Board of Commissioners for Foreign Missions missionary Joseph Travelli, who had him enrolled in a Chinese day school established by his colleague Ira Tracy the previous year. The school's missionaries referred to him as \"Chan Laisun\".Around April 1843, Zeng was sent to the United States to continue his education. He was accompanied by the Presbyterian missionary John Hunter Morrison, who was returning to America after work in northern India. They went west via the Indian and Atlantic oceans, sailing around the Cape of Good Hope to dock on the east coast of the country. Morrison raised funds from among his friends and enrolled Zeng in the Bloomfield Academy, a boarding school in Bloomfield, New Jersey. Morrison returned to India in 1846, and Zeng was put into the care of an American Board missionary previously stationed in Guangzhou, Samuel Wells Williams. In the fall of 1846, Zeng transferred from Bloomfield to Hamilton College, a Presbyterian institution in Clinton, New York. Williams arranged for the First Presbyterian Church in Utica, New York, to support Zeng's study for two years, the \"faculty offering to teach him gratuitously, and the ladies in Brooklyn to clothe him.\".According to historian Edward J. M. Rhoads, Zeng was the first Chinese person to attend college in the United States, and possibly the first at any foreign college. At Hamilton, Zeng studied the New Testament in Koine Greek (likely under classicist Edward North) and taught Sunday school at a local church. He was active in the college's glee club. In early 1848, his funding ran out, and he was forced to withdraw from the college. Williams attempted to arrange for the American Board to take Zeng to China to work as a teacher at their mission in Xiamen, but the American Board refused, stating that it was the Presbyterian Church's responsibility to transport him, and that foreign-educated Chinese Christians were unsuitable for mission work. Instead, he traveled to China with Williams and his wife, departing from New York City in late May 1848."],
            "fixed_list_repeat":5000,
            "hf_dataset_name": "wikitext",
            "hf_dataset_config_name": "wikitext-103-raw-v1", # 使用小数据集
            "hf_split": "train[:5%]", # 只用训练集的前10%
            "hf_text_column": "text",
            "max_samples_to_load": 1,  # 大幅减少样本量，增加重复性
            "min_raw_text_length": 128,    # 筛选长度适中的文本
            "max_raw_text_length": 4096,    # 原始文本的最大字符长度，用于筛选更同质化的短文本
            "min_tokenized_length": 128,
            "max_tokenized_length": 256,   # 使用较短且固定的分块长度
            "stride_for_chunking": 128,     # 较大的重叠，增加数据片段间的相似性
            "prompt_min_len_ratio": 0.3,   # prompt比例可以稍大，让上下文更固定
            "prompt_max_len_ratio": 0.5,
        },
        "training_params_override": { # 覆盖 TRAINING_PARAMS 的参数
            "COMPRESSOR_LEARNING_RATE": 0.00001, # 第一阶段可以使用稍高或不同的学习率
            "KL_TEMPERATURE": 2,           # 可以用稍高的温度使目标分布更平滑
            "MAX_TOKENS_PER_EPISODE": 5,    # 每个文本块续写的token数可以少一些
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
    "model_name_or_path": "/raid_sdh/home/xyg/PRETRAINED_MODEL/TinyLlama",
    "tokenizer_name_or_path": "/raid_sdh/home/xyg/PRETRAINED_MODEL/TinyLlama",
    "device": "cuda:5" if torch.cuda.is_available() else "cpu",
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