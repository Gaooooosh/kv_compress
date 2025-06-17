import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationConfig
from transformers.cache_utils import DynamicCache
from typing import Optional

# 假设这些模块可以从您的项目中导入
from model import KVCompressor
from untils import CompressorConfig
from CCache import CacheProcessor


class CausalLMWithKVCompressor:
    """
    一个包装类，为标准的Hugging Face CausalLM模型增加了KV缓存压缩功能。

    这个类封装了一个预训练的LLM，并重载了其`generate`方法，以实现
    “压缩预填充”策略。它首先计算整个输入提示的KV缓存，然后使用
    预训练的KVCompressor对其进行分块压缩，最后将压缩后的缓存作为
    上下文启动自回归文本生成。
    """
    def __init__(self, model: PreTrainedModel, compressor_config: CompressorConfig):
        """
        初始化包装类。

        Args:
            model (PreTrainedModel): 一个已加载的、预训练的Hugging Face CausalLM模型实例。
            compressor_config (CompressorConfig): KV压缩器的配置对象。
        """
        self.model = model
        self.compressor_config = compressor_config
        self.device = self.model.device

        # 初始化K和V压缩器，并将其转换到正确的设备和数据类型
        self.k_compressor = KVCompressor(self.compressor_config).to(dtype=self.dtype)
        self.v_compressor = KVCompressor(self.compressor_config).to(dtype=self.dtype)
        self.k_compressor.eval()
        self.v_compressor.eval()

        self.processor = CacheProcessor(self.compressor_config)

    def load_compressors(self, checkpoint_path: str):
        """
        从单个检查点文件 (.pth) 加载K和V压缩器的权重。

        Args:
            checkpoint_path (str): 包含压缩器状态字典的检查点文件路径。
        """
        print(f"Loading K and V compressor weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if 'k_compressor_state_dict' in checkpoint:
            self.k_compressor.load_state_dict(checkpoint['k_compressor_state_dict'])
            print("Successfully loaded k_compressor_state_dict.")
        else:
            raise KeyError("Checkpoint missing 'k_compressor_state_dict'.")

        if 'v_compressor_state_dict' in checkpoint:
            self.v_compressor.load_state_dict(checkpoint['v_compressor_state_dict'])
            print("Successfully loaded v_compressor_state_dict.")
        else:
            # 如果检查点中没有V压缩器的权重，则默认让V和K共享权重
            print("Warning: 'v_compressor_state_dict' not found. V-Compressor will share weights with K-Compressor.")
            self.v_compressor.load_state_dict(checkpoint['k_compressor_state_dict'])
            
    def __getattr__(self, name: str):
        """
        将所有其他属性和方法的调用委托给内部的原始LLM模型。
        """
        return getattr(self.model, name)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        prefill_chunk_size: int,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        重载的generate方法，实现了压缩预填充和完整的手动自回归生成循环。

        Args:
            input_ids (torch.Tensor): 输入提示的token ID。
            prefill_chunk_size (int): 在预填充阶段，对KV缓存进行分块压缩的块大小。
            generation_config (Optional[GenerationConfig], optional): Hugging Face的生成配置。
            **kwargs: 其他可以传递给 model.generate 的参数 (在此手动循环中被部分使用)。

        Returns:
            torch.Tensor: 生成的token ID序列，包含原始输入。
        """
        
        # --- 1. 预填充（Prefill）阶段 ---
        # 这个阶段的逻辑已经正确，保持不变
        print("Step 1: Performing full forward pass to get uncompressed KV cache...")
        uncompressed_past_key_values = self.model(input_ids=input_ids, use_cache=True, return_dict=True).past_key_values

        print(f"Step 2: Compressing the cache layer by layer with chunk size {prefill_chunk_size}...")
        final_compressed_past = []
        for i in range(self.model.config.num_hidden_layers):
            k_tensor_layer_i, v_tensor_layer_i = uncompressed_past_key_values[i]
            k_chunks, v_chunks = list(torch.split(k_tensor_layer_i, prefill_chunk_size, dim=2)), list(torch.split(v_tensor_layer_i, prefill_chunk_size, dim=2))
            compressed_k_chunks, compressed_v_chunks = [], []
            for k_chunk, v_chunk in zip(k_chunks, v_chunks):
                target_device, target_dtype = k_chunk.device, k_chunk.dtype
                self.k_compressor.to(target_device, dtype=target_dtype)
                self.v_compressor.to(target_device, dtype=target_dtype)
                prepared_k = self.processor.prepare_segments_for_global_compression([k_chunk], k_chunk.shape[2])
                prepared_v = self.processor.prepare_segments_for_global_compression([v_chunk], v_chunk.shape[2])
                compressed_k, compressed_v = self.k_compressor(prepared_k), self.v_compressor(prepared_v)
                formatted_k, formatted_v = self.processor.format_globally_compressed_output(compressed_k, 1)[0], self.processor.format_globally_compressed_output(compressed_v, 1)[0]
                compressed_k_chunks.append(formatted_k)
                compressed_v_chunks.append(formatted_v)
            final_compressed_past.append((torch.cat(compressed_k_chunks, dim=2), torch.cat(compressed_v_chunks, dim=2)))
        
        prefilled_cache_tuple = tuple(final_compressed_past)
        compressed_len = prefilled_cache_tuple[0][0].shape[2]
        print(f"Step 3: Prefill complete. Compressed cache length: {compressed_len}.")

        # --- 2. 手动自回归生成循环 (修正版) ---
        print("Step 4: Starting auto-regressive generation with manual loop...")

        if generation_config is None:
            generation_config = self.model.generation_config
        for key, value in kwargs.items():
            setattr(generation_config, key, value)
        
        # --- 关键修改 1: 将初始缓存元组转换为DynamicCache对象 ---
        past_key_values = DynamicCache.from_legacy_cache(prefilled_cache_tuple)

        next_tokens = input_ids[:, -1:]
        generated_ids = []
        
        for _ in range(generation_config.max_new_tokens):
            # 直接使用模型核心的 forward 方法
            outputs = self.model(
                input_ids=next_tokens,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]

            if getattr(generation_config, 'do_sample', False):
                 if getattr(generation_config, 'temperature', 1.0) > 0:
                     next_token_logits = next_token_logits / generation_config.temperature
                 if getattr(generation_config, 'top_k', 0) > 0:
                    top_k = generation_config.top_k
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                 probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                 next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                 next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids.append(next_tokens)

            # --- 关键修改 2: 每次循环后，都将返回的元组缓存转换回DynamicCache对象 ---
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

            if generation_config.eos_token_id is not None and (next_tokens == generation_config.eos_token_id).all():
                break
        
        if len(generated_ids) > 0:
            generated_tensor = torch.cat(generated_ids, dim=1)
            return torch.cat([input_ids, generated_tensor], dim=-1)
        else:
            return input_ids

    
if __name__ == '__main__':
    # --- 1. 定义路径 ---
    llm_path = "/raid_sdh/home/xyg/PRETRAINED_MODEL/TinyLlama"
    # 现在只需要一个检查点路径
    compressor_checkpoint_path = "/raid_sdh/home/xyg/compressor_training_output_dataset/kv_compressor_alt_final_overall.pth"

    # --- 2. 加载基础LLM和分词器 ---
    print("Loading base LLM and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 3. 定义压缩器配置 ---
    llm_true_config = base_model.config
    cfg_kv_heads = getattr(llm_true_config, 'num_key_value_heads', llm_true_config.num_attention_heads)
    cfg_input_dim = (llm_true_config.hidden_size // llm_true_config.num_attention_heads) * cfg_kv_heads
    
    compressor_config_for_inference = CompressorConfig(
        input_dim=cfg_input_dim,
        reduction_factor=4,
        output_seq_len=4,
        num_attention_heads=32,
        use_mixed_precision=False,
        torch_dtype="torch.float32",
        kv_head_dim=cfg_kv_heads,
        layer_nums=llm_true_config.num_hidden_layers,
        kernel_size = 3,
        padding = 1,
        compression_threshold = 64,
        initial_uncompressed_keep_length = 8
    )
    
    # --- 4. 实例化包装类并加载压缩器权重 ---
    print("Instantiating the CausalLMWithKVCompressor...")
    compressed_model = CausalLMWithKVCompressor(
        model=base_model,
        compressor_config=compressor_config_for_inference
    )
    # 从单个文件加载权重
    compressed_model.load_compressors(compressor_checkpoint_path)

    # --- 5. 准备输入并调用generate方法 ---
    input_prompt = "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope. This will enable investigations in many fields of astronomy and cosmology, such as observation of the first stars and the formation of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets. So, what can it do for human?"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(base_model.device)

    final_output_ids = compressed_model.generate(
        input_ids=input_ids,
        prefill_chunk_size=32, # 定义每个压缩块的原始长度
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    
    generated_text = tokenizer.decode(final_output_ids[0], skip_special_tokens=True)

    print("\n--- Final Generated Text ---")
    print(generated_text)