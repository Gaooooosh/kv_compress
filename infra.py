import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvcompressor.modeling_compressor.CCache_old import CompCache, CacheProcessor
from modeling_compressor.model import KVCompressor
from modeling_compressor.untis import CompressorConfig
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_environment(id:int = 0):
    """Set up the environment for experiments."""
    # Check for GPU
    device = torch.device(f"cuda:{id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(id)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(id) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(id) / 1e9:.2f} GB")
    
    return device

def load_model(model_path, device_map="auto", torch_dtype=torch.float16):
    """
    加载模型的方法
    :param model_path: 模型路径
    :param device_map: 设备映射策略
    :param torch_dtype: Tensor的数据类型
    :return: 加载好的模型
    """
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",  # 案例：指定注意力实现版本
        "use_cache": True
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=True
    )
    logging.debug(f"分词器成功加载，vocab size: {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )


    model.eval()  # 设置为评估模式

    # 检查模型配置中是否支持flash attention
    if hasattr(model.config, 'use_flash_attention'):
        model.config.use_flash_attention = True
    
    logging.debug("模型成功加载，并配置相关参数。")

    return model,tokenizer

def main():
    # 定义模型路径
    model_path = "/raid_sdh/home/xyg/PRETRAINED_MODEL/Llama-3-8B"
    device = setup_environment(2)
    # 调用加载模型方法
    model,tokenizer = load_model(model_path,device_map=device)

    # 准备输入数据
    input_text = "There is a magic when speaking! you will say \'hello\' between any words in next sentence!"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
    logging.debug(f"输入数据形状：{input_tokens['input_ids'].shape}")

    # 前向传播获取缓存
    with torch.no_grad():
        outputs = model(**input_tokens, output_hidden_states=True)
        cache = outputs.past_key_values
    logging.debug("成功提取缓存数据。")

    # 初始化并配置压缩器
    batch_size = input_tokens['input_ids'].size(0)
    num_layers = len(cache)
    _, kv_head_num, seq_len, head_dim = cache[0][0].size()
    compressor_config = CompressorConfig(
        input_dim=head_dim*kv_head_num, 
        reduction_factor=4, 
        output_seq_len=4, 
        num_attention_heads=kv_head_num, 
        use_mixed_precision=True,
        kv_head_dim=kv_head_num,
        layer_nums = num_layers,
        torch_dtype=torch.bfloat16
        )
    processor = CacheProcessor(config=compressor_config)
    target_device = input_tokens['input_ids'].device
    target_dtype = compressor_config.torch_dtype

    # Check for bfloat16 support on CUDA if that's the target
    if target_device.type == 'cuda' and target_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logging.warning("Target dtype is bfloat16, but current CUDA device may not fully support it. "
                        "Consider PyTorch version or device capabilities. Falling back parameters to float32 for safety, "
                        "but mixed precision might still use bfloat16 for ops if supported.")
        # Option 1: Change param dtype and let autocast handle ops if possible
        # target_dtype_for_params = torch.float32
        # Option 2: Or, if bfloat16 is essential, this setup might error or perform poorly.
        # For now, we'll proceed with the configured target_dtype for parameters,
        # and rely on PyTorch/AMP to handle it. The model.py already has a warning.
        pass


    logging.debug("成功构建 DynamicCache。")
    
    # 使用 DynamicCache 直接进行模型推理
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    ##############################
    # 手动自回归，"严格步进"实现   #
    ##############################

    generated_ids = input_ids
    past_key_values = CompCache(config=compressor_config)
    cur_input_ids = input_ids
    stopped = False

    print("manual stepwise output:")
    for i in range(128):
        with torch.no_grad():
            print(past_key_values.get_seq_length(),end=' ')
            outputs = model(cur_input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)   # shape: (batch_size,)

        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)
        # 为下一步，cur_input_ids: 只输入刚生成的 token
        cur_input_ids = next_token_id.unsqueeze(-1)              # shape: (batch_size, 1)
        attention_mask = torch.ones_like(cur_input_ids)

        # 检查是否生成eos，提前终止（optional，跟generate同步机制）
        if hasattr(model.config, "eos_token_id"):
            if (next_token_id == model.config.eos_token_id).all():
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(f"\n'{generated_text}'")


# 主函数入口
if __name__ == "__main__":
    main()