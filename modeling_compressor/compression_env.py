# compression_env.py (Modified for Simplified Global CompCache)
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import random
from datasets import load_dataset, Dataset
from CCache import CompCache # 使用简化版的 CompCache
from untis import CompressorConfig
from config_rl import TRAINING_PARAMS, LLM_CONFIG

class CompressionEnv:
    def __init__(self,
                 compressor_config: CompressorConfig,
                 llm_model_name_or_path: str,
                 tokenizer_name_or_path: str,
                 llm_device: str,
                 training_params: Dict[str, Any] = TRAINING_PARAMS,
                 llm_config_params: Dict[str, Any] = LLM_CONFIG):
        print("Initializing Compression Environment with Dataset Support...")
        self.training_params = training_params
        self.llm_config_params = llm_config_params
        self.dataset_args = training_params.get("dataset_args", {}) # 获取数据集配置
        self.device = torch.device(llm_device)

        print(f"Loading Tokenizer: {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading Main LLM Model: {llm_model_name_or_path} to device: {self.device}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
        self.llm_model.to(self.device)
        self.llm_model.eval()
        print("Main LLM Model loaded.")

        # 动态调整CompressorConfig (与之前版本类似)
        # ... (这部分基本不变，确保从llm_model.config获取正确的维度信息) ...
        llm_num_layers = self.llm_model.config.num_hidden_layers
        llm_num_heads = self.llm_model.config.num_attention_heads
        llm_num_kv_heads = getattr(self.llm_model.config, 'num_key_value_heads', llm_num_heads)
        llm_hidden_size = self.llm_model.config.hidden_size
        llm_head_dim = llm_hidden_size // llm_num_heads
        expected_compressor_input_dim = llm_num_kv_heads * llm_head_dim

        original_comp_layer_ids = compressor_config.compress_layer_ids
        compressor_config.layer_nums = llm_num_layers
        if compressor_config.input_dim != expected_compressor_input_dim:
            compressor_config.input_dim = expected_compressor_input_dim
        if compressor_config.kv_head_dim != llm_num_kv_heads:
            compressor_config.kv_head_dim = llm_num_kv_heads
        
        valid_compress_layer_ids = []
        if original_comp_layer_ids:
            for layer_id in original_comp_layer_ids:
                if 0 <= layer_id < llm_num_layers: valid_compress_layer_ids.append(layer_id)
        compressor_config.compress_layer_ids = valid_compress_layer_ids
        
        self.comp_cache = CompCache(config=compressor_config, device=self.device)
        print("CompCache initialized.")

        # 加载和预处理数据集
        self.dataset: Optional[Dataset] = None
        self._load_and_prepare_dataset()

        self._current_text_segment_token_ids: List[int] = [] # 当前正在处理的文本段的token ids
        self._current_segment_processed_tokens = 0 # 在当前文本段中已处理（作为LLM输入）的token数
        self._past_key_values_ref: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None
        self._current_episode_total_tokens_generated = 0 # 当前“回合”LLM新生成的token数 (用于done判断)

        self.compression_trigger_threshold = compressor_config.compression_threshold
        self.initial_keep_len = compressor_config.initial_uncompressed_keep_length
        print("Compression Environment initialized successfully with dataset.")


    def _load_and_prepare_dataset(self):
        """加载并准备数据集"""
        dataset_name = self.dataset_args.get("dataset_name")
        if not dataset_name:
            print("No dataset_name provided in config. Environment will use a default short prompt for reset.")
            self.dataset = None
            return

        print(f"Loading dataset: {dataset_name}...")
        try:
            # 加载数据集，可以取一部分样本用于快速迭代
            num_samples = self.dataset_args.get("max_samples_to_load", None)
            split_arg = self.dataset_args.get("split", "train")
            if num_samples:
                split_arg = f"{split_arg}[:{num_samples}]" # e.g., "train[:1000]"

            self.dataset = load_dataset(
                dataset_name,
                self.dataset_args.get("dataset_config_name"),
                split=split_arg,
                # streaming=True, # 可以考虑流式加载大型数据集
            )
            self.dataset = self.dataset.filter(lambda example: len(example[self.dataset_args["text_column"]]) > self.dataset_args.get("min_text_length_for_sample", 50)) # 过滤掉太短的文本
            self.dataset = self.dataset.shuffle(seed=TRAINING_PARAMS.get("RANDOM_SEED", 42)) # 打乱数据集
            self.dataset_iterator = iter(self.dataset) # 创建迭代器
            print(f"Dataset '{dataset_name}' loaded. Number of samples (approx): {len(self.dataset) if hasattr(self.dataset, '__len__') else 'Streaming'}")
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}. Will use default prompt.")
            self.dataset = None


    def _get_next_text_segment_from_dataset(self) -> List[int]:
        """从数据集中获取下一个文本段并tokenize"""
        if not self.dataset or not self.dataset_iterator:
            # print("Dataset not available or exhausted, using default prompt.")
            # Fallback to a very short, fixed prompt if dataset fails or ends
            default_prompt = "This is a default sentence for the language model."
            return self.tokenizer(default_prompt, return_tensors="pt").input_ids[0].tolist()

        text_col = self.dataset_args["text_column"]
        max_len = self.dataset_args.get("max_text_length_for_sample", 
                                        self.training_params.get("IDEAL_TOTAL_CONTEXT_LENGTH", 512))
        
        try:
            example = next(self.dataset_iterator)
            text = example[text_col]
            # 清理文本，移除过多换行等 (可选)
            text = " ".join(text.split()) 
            token_ids = self.tokenizer(text, truncation=True, max_length=max_len).input_ids
            # print(f"  Loaded segment from dataset, length: {len(token_ids)}")
            return token_ids
        except StopIteration:
            print("Dataset iterator exhausted. Re-shuffling and creating new iterator.")
            self.dataset = self.dataset.shuffle(seed=TRAINING_PARAMS.get("RANDOM_SEED", 42) + 1) # 用不同的种子再次打乱
            self.dataset_iterator = iter(self.dataset)
            try:
                example = next(self.dataset_iterator)
                text = example[text_col]
                text = " ".join(text.split())
                token_ids = self.tokenizer(text, truncation=True, max_length=max_len).input_ids
                return token_ids
            except StopIteration: # 如果数据集实在太小
                print("Error: Dataset exhausted even after reshuffle. Using default prompt.")
                default_prompt = "This is a default sentence after dataset exhaustion."
                return self.tokenizer(default_prompt, return_tensors="pt").input_ids[0].tolist()



    def _get_initial_prompt_ids(self, batch_size: int = 1) -> torch.Tensor: # (保持不变)
        initial_text = "User: Hello, let's discuss advanced AI topics.\nAgent:"
        inputs = self.tokenizer(initial_text, return_tensors="pt", padding=False, truncation=True, 
                                max_length=self.llm_config_params["max_context_length_for_llm_input"] // 4)
        return inputs.input_ids.to(self.device)

    def reset(self) -> None:
        """重置环境，加载新的文本段作为上下文"""
        self.comp_cache._initialize_cache_lists() # 清空并重置 CompCache
        self._past_key_values_ref = None # 重置参考路径的KV缓存

        self._current_text_segment_token_ids = self._get_next_text_segment_from_dataset()
        self._current_segment_processed_tokens = 0 # 在当前段中已作为LLM输入的token数
        self._current_episode_total_tokens_generated = 0 # 新生成的token计数器重置

        # 初始上下文长度，用于预填充（例如，取文本段的一半或固定长度）
        # IDEAL_TOTAL_CONTEXT_LENGTH 是LLM期望的最大长度，但初始prompt可以短一些
        prompt_len_initial = min(len(self._current_text_segment_token_ids) // 2, 
                                 self.llm_config_params["max_context_length_for_llm_input"] // 2)
        prompt_len_initial = max(1, prompt_len_initial) # 至少一个token

        initial_input_ids_list = self._current_text_segment_token_ids[:prompt_len_initial]
        initial_input_ids = torch.tensor([initial_input_ids_list], dtype=torch.long, device=self.device)
        self._current_segment_processed_tokens = prompt_len_initial
        
        # print(f"Env Reset: Loaded new text segment. Initial prompt length: {prompt_len_initial}")

        if initial_input_ids.shape[1] == 0: # 如果初始prompt为空（例如文本段太短）
            # print("Warning: Initial prompt for reset is empty. Using a single BOS token.")
            initial_input_ids = torch.tensor([[self.tokenizer.bos_token_id or 0]], dtype=torch.long, device=self.device)
            self._current_segment_processed_tokens = 1
            if not self._current_text_segment_token_ids: # 确保列表非空
                 self._current_text_segment_token_ids = initial_input_ids[0].tolist()


        attention_mask = torch.ones_like(initial_input_ids)
        with torch.no_grad():
            outputs_main = self.llm_model(
                input_ids=initial_input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True, return_dict=True
            )
            if outputs_main.past_key_values:
                self.comp_cache.update_from_hf_past(outputs_main.past_key_values)
            
            outputs_ref = self.llm_model(
                input_ids=initial_input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True, return_dict=True
            )
            self._past_key_values_ref = outputs_ref.past_key_values
        
    def _calculate_loss(self, logits_compressed: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor: # (保持不变)
        loss_type = self.training_params.get("LOSS_FUNCTION", "KL").upper()
        logits_ref_detached = logits_ref.detach()
        if loss_type == "KL":
            kl_temp = self.training_params.get("KL_TEMPERATURE", 1.0)
            log_probs_compressed = F.log_softmax(logits_compressed / kl_temp, dim=-1)
            probs_ref = F.softmax(logits_ref_detached / kl_temp, dim=-1)
            loss = F.kl_div(log_probs_compressed, probs_ref, reduction='batchmean', log_target=False)
        elif loss_type == "MSE":
            loss = F.mse_loss(logits_compressed, logits_ref_detached)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_type}")
        return loss

    def _calculate_logits_entropy(self, logits: Optional[torch.Tensor]) -> float: # (保持不变)
        if logits is None or logits.numel() == 0: return -1.0
        probs = F.softmax(logits.detach(), dim=-1)
        log_probs = F.log_softmax(logits.detach(), dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()

    def generate_one_step(self) -> Optional[Tuple[torch.Tensor, bool, Dict[str, Any]]]:
        """环境执行一步生成，返回损失、是否结束、信息字典"""
        if self._current_segment_processed_tokens >= len(self._current_text_segment_token_ids):
            # print("Current text segment exhausted in env.generate_one_step. Resetting.")
            self.reset() # 如果当前文本段已处理完，则加载新的
            if self._current_segment_processed_tokens >= len(self._current_text_segment_token_ids): # 再次检查，防止空数据集
                # print("ERROR: Dataset seems to provide empty segments consistently after reset.")
                return torch.tensor(0.0, device=self.device, requires_grad=False), True, {"error": "Empty segment after reset"}


        # 获取当前LLM的输入token (通常是上一个生成的token，或者文本段中的下一个真实token)
        # 为了与参考路径对齐，我们从 self._current_text_segment_token_ids 中取下一个token作为输入
        current_input_token_id = self._current_text_segment_token_ids[self._current_segment_processed_tokens]
        current_input_ids = torch.tensor([[current_input_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(current_input_ids)

        # --- 主LLM (使用CompCache) ---
        logits_compressed: Optional[torch.Tensor] = None
        
        try:
            with torch.set_grad_enabled(True): 
                self.llm_model.eval()
                for param in self.comp_cache.k_compressor.parameters(): param.requires_grad = True
                for param in self.comp_cache.v_compressor.parameters(): param.requires_grad = True

                outputs_main = self.llm_model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache,
                    use_cache=True, return_dict=True
                )
                logits_compressed = outputs_main.logits[:, -1, :] # Logits for the token *after* current_input_token_id
        except Exception as e:
            print(f"Error during Main LLM forward: {e}")
            return None, True, {"error": str(e)}


        # --- 参考LLM (无压缩) ---
        logits_ref: Optional[torch.Tensor] = None
        try:
            with torch.no_grad():
                self.llm_model.eval()
                outputs_ref = self.llm_model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self._past_key_values_ref,
                    use_cache=True, return_dict=True
                )
                logits_ref = outputs_ref.logits[:, -1, :]
                self._past_key_values_ref = outputs_ref.past_key_values
        except Exception as e:
            print(f"Error during Reference LLM forward: {e}")
            return None, True, {"error_ref": str(e)}

        # 更新已处理的token计数和总生成token计数
        self._current_segment_processed_tokens += 1
        self._current_episode_total_tokens_generated += 1 # LLM尝试生成了一个新token

        # --- 固定规则触发全局压缩 ---
        compression_triggered_this_step = False
        compression_performed_this_step = False
        # （压缩触发逻辑与之前版本类似，检查 compress_layer_ids 中是否有层达到阈值）
        for layer_idx_check in self.comp_cache.config.compress_layer_ids:
            if layer_idx_check < len(self.comp_cache.key_cache) and \
               self.comp_cache.key_cache[layer_idx_check].ndim == 4: # 确保层缓存已初始化
                layer_total_len = self.comp_cache.key_cache[layer_idx_check].shape[-2]
                if (layer_total_len - self.initial_keep_len) >= self.compression_trigger_threshold:
                    compression_triggered_this_step = True
                    break
        if compression_triggered_this_step:
            compression_performed_this_step = self.comp_cache.trigger_global_compression()

        # --- 计算损失 ---
        loss = torch.tensor(0.0, device=self.device) 
        if logits_compressed is not None and logits_ref is not None:
            loss = self._calculate_loss(logits_compressed, logits_ref)
        else: # Logits 获取失败
             return torch.tensor(10.0, device=self.device, requires_grad=True), True, {"error": "Logits missing"} # 返回一个大损失并结束


        done = self._current_episode_total_tokens_generated >= self.training_params["MAX_TOKENS_PER_EPISODE"]
        
        # 用argmax获取预测的token，仅用于info日志
        predicted_token_main = torch.argmax(logits_compressed.detach(), dim=-1).item() if logits_compressed is not None else -1
        predicted_token_ref = torch.argmax(logits_ref.detach(), dim=-1).item() if logits_ref is not None else -1
        
        info = {
            "loss_value": loss.item(),
            "compressed_logits_entropy": self._calculate_logits_entropy(logits_compressed),
            "ref_logits_entropy": self._calculate_logits_entropy(logits_ref),
            "compression_triggered_this_step": compression_triggered_this_step,
            "compression_performed_this_step": compression_performed_this_step,
            "predicted_token_main": predicted_token_main,
            "predicted_token_ref": predicted_token_ref,
            "processed_tokens_in_segment": self._current_segment_processed_tokens,
            "total_tokens_in_segment": len(self._current_text_segment_token_ids)
        }
        return loss, done, info

    def _get_simplified_cache_stats(self, layer_idx: int = 0) -> Dict[str, int]:
        """获取指定层简化后的缓存统计（总长度）用于info"""
        if layer_idx < len(self.comp_cache.key_cache) and \
           self.comp_cache.key_cache[layer_idx].ndim == 4 :
            total_len = self.comp_cache.key_cache[layer_idx].shape[-2]
            # 由于移除了精细的 compressed_len 和 pending_len 跟踪，这里只能提供总长
            # 如果需要更详细的，CompCache内部需要暴露更多信息或重新引入部分状态跟踪
            return {"total_len": total_len}
        return {"total_len": 0}

    def close(self): # (保持不变)
        print("Closing Compression Environment.")
        del self.llm_model; del self.tokenizer; del self.comp_cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__': # (测试脚本也需要相应调整以使用新的数据集加载逻辑)
    print("Testing Compression Environment with Dataset Support...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    from config_rl import DEFAULT_COMPRESSOR_CONFIG_PARAMS as DCCP # 确保这个在config_rl.py定义了
    # 为测试创建一个CompressorConfig
    dataset_args_test = TRAINING_PARAMS.get("dataset_args", {})
    dataset_args_test.setdefault("dataset_name", "wikitext") # 使用wikitext进行测试
    dataset_args_test.setdefault("dataset_config_name", "wikitext-2-raw-v1") # 用小一点的wikitext-2
    dataset_args_test.setdefault("text_column", "text")
    dataset_args_test.setdefault("split", "train")
    dataset_args_test.setdefault("max_samples_to_load", 10) # 只加载少量样本测试
    dataset_args_test.setdefault("min_text_length_for_sample", 64)
    dataset_args_test.setdefault("max_text_length_for_sample", 256)
    
    current_training_params = TRAINING_PARAMS.copy()
    current_training_params["dataset_args"] = dataset_args_test
    current_training_params["MAX_TOKENS_PER_EPISODE"] = 50 # 测试时少生成一些


    test_env_compressor_config = CompressorConfig(
        input_dim=DCCP.get("input_dim", 128),
        reduction_factor=DCCP.get("reduction_factor", 2),
        output_seq_len=DCCP.get("output_seq_len", 4),
        num_attention_heads=DCCP.get("num_attention_heads", 2),
        use_mixed_precision=False, torch_dtype=torch.float32,
        kv_head_dim=DCCP.get("kv_head_dim", 2),
        layer_nums=DCCP.get("layer_nums", 2), # 会被LLM真实层数覆盖
        kernel_size=DCCP.get("kernel_size", 3), padding=DCCP.get("padding", 1),
        compression_threshold=DCCP.get("compression_threshold", 8),
        initial_uncompressed_keep_length=DCCP.get("initial_uncompressed_keep_length", 2),
        compress_layer_ids=DCCP.get("compress_layer_ids", [0,1]) # 会被env动态调整
    )

    try:
        env = CompressionEnv(
            compressor_config=test_env_compressor_config,
            llm_model_name_or_path=LLM_CONFIG["model_name_or_path"], # "gpt2" for small test
            tokenizer_name_or_path=LLM_CONFIG.get("tokenizer_name_or_path", LLM_CONFIG["model_name_or_path"]),
            llm_device=LLM_CONFIG["device"],
            training_params=current_training_params,
            llm_config_params=LLM_CONFIG
        )
    except Exception as e:
        print(f"Error during CompressionEnv test initialization: {e}")
        raise

    num_test_episodes_env = 2
    for ep in range(num_test_episodes_env):
        print(f"\n--- Test Episode {ep + 1}/{num_test_episodes_env} ---")
        env.reset() # 加载新的文本段
        for step in range(TRAINING_PARAMS["MAX_TOKENS_PER_EPISODE"]): # 根据配置的每回合最大生成token数
            print(f"\n  Episode {ep+1}, Step {step + 1}/{TRAINING_PARAMS['MAX_TOKENS_PER_EPISODE']}")
            result = env.generate_one_step()
            if result is None:
                print("    generate_one_step returned None, breaking episode.")
                break
            loss_tensor, done, info = result
            if loss_tensor is not None:
                print(f"    Loss: {loss_tensor.item():.6f}")
                print(f"    Info: {info}")
            else:
                print("    Loss tensor is None.")
            if done:
                print(f"  Episode finished at step {step + 1}.")
                break
    env.close()
    print("\nCompression Environment with Dataset test finished.")