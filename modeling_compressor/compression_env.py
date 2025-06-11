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
from untils import CompressorConfig
from config_rl import TRAINING_PARAMS, LLM_CONFIG

class CompressionEnv:
    def __init__(self,
                 compressor_config: CompressorConfig,
                 llm_model_name_or_path: str,
                 tokenizer_name_or_path: str,
                 llm_device: str,
                 training_params: Dict[str, Any] = TRAINING_PARAMS,
                 llm_config_params: Dict[str, Any] = LLM_CONFIG,
                 current_stage_dataset_args: Optional[Dict[str, Any]] = None):
        print("Initializing Compression Environment (Revised for Gradient Flow)...")
        self.training_params = training_params
        self.llm_config_params = llm_config_params
        self.dataset_args = current_stage_dataset_args if current_stage_dataset_args is not None \
                            else training_params.get("dataset_args", {})
        self.device = torch.device(llm_device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
        self.llm_model.to(self.device)
        self.llm_model.eval()
        print("Main LLM Model loaded.")

        llm_num_layers = self.llm_model.config.num_hidden_layers
        llm_num_heads = self.llm_model.config.num_attention_heads
        llm_num_kv_heads = getattr(self.llm_model.config, 'num_key_value_heads', llm_num_heads)
        llm_hidden_size = self.llm_model.config.hidden_size
        llm_head_dim = llm_hidden_size // llm_num_heads
        expected_compressor_input_dim = llm_num_kv_heads * llm_head_dim

        original_comp_layer_ids = compressor_config.compress_layer_ids
        compressor_config.layer_nums = llm_num_layers
        if not hasattr(compressor_config, 'input_dim') or compressor_config.input_dim is None or compressor_config.input_dim != expected_compressor_input_dim :
            # print(f"Info: Overriding CompressorConfig.input_dim from {compressor_config.input_dim} to LLM's expected {expected_compressor_input_dim}.")
            compressor_config.input_dim = expected_compressor_input_dim
        if not hasattr(compressor_config, 'kv_head_dim') or compressor_config.kv_head_dim is None or compressor_config.kv_head_dim != llm_num_kv_heads:
            # print(f"Info: Overriding CompressorConfig.kv_head_dim from {compressor_config.kv_head_dim} to LLM's {llm_num_kv_heads}.")
            compressor_config.kv_head_dim = llm_num_kv_heads
        
        valid_compress_layer_ids = []
        if original_comp_layer_ids:
            for layer_id in original_comp_layer_ids:
                if 0 <= layer_id < llm_num_layers: valid_compress_layer_ids.append(layer_id)
        compressor_config.compress_layer_ids = valid_compress_layer_ids
        
        self.comp_cache = CompCache(config=compressor_config, device=self.device)
        print("CompCache initialized.")

        self.all_tokenized_chunks: List[List[int]] = []
        self.dataset_chunk_iterator: Optional[iter] = None
        self._load_and_prepare_dataset()

        self._current_text_segment_token_ids: List[int] = []
        self._current_segment_processed_tokens = 0
        self._past_key_values_ref = None
        self._current_episode_total_tokens_generated = 0
        
        self.compression_trigger_threshold = compressor_config.compression_threshold
        self.initial_keep_len = compressor_config.initial_uncompressed_keep_length
        
        self.ema_alpha = self.training_params.get("EMA_ALPHA_LOGITS_REF", 0.1)
        self.ema_logits_ref: Optional[torch.Tensor] = None
        print(f"Using EMA for reference logits with alpha: {self.ema_alpha}")
        print(f"Compression Environment initialized for dataset args: {self.dataset_args.get('hf_dataset_name', 'default prompt')}")


    def _load_and_prepare_dataset(self): # (与您之前能成功加载数据的版本一致)
        source_type = self.dataset_args.get("source_type", "huggingface_dataset")
        self.all_tokenized_chunks = [] 
        if source_type == "huggingface_dataset":
            dataset_name = self.dataset_args.get("hf_dataset_name")
            if not dataset_name:
                print("No hf_dataset_name provided. Using default prompt for data generation.")
            else:
                print(f"Loading HF dataset: {dataset_name} with config: {self.dataset_args.get('hf_dataset_config_name')}")
                try:
                    num_samples_str = str(self.dataset_args.get("max_samples_to_load", ""))
                    split_arg = self.dataset_args.get("hf_split", "train")
                    if num_samples_str and ":" not in split_arg and "%" not in split_arg: # 避免重复切片
                        split_arg = f"{split_arg}[:{num_samples_str}]"
                    
                    raw_dataset = load_dataset(
                        dataset_name,
                        self.dataset_args.get("hf_dataset_config_name"),
                        split=split_arg,
                    )
                    print(f"  Initial raw dataset size from split '{split_arg}': {len(raw_dataset)}")

                    min_raw_len = self.dataset_args.get("min_raw_text_length", 0)
                    max_raw_len = self.dataset_args.get("max_raw_text_length", float('inf'))
                    text_col = self.dataset_args.get("hf_text_column", "text") # 使用get并提供默认值
                    
                    if min_raw_len > 0 or max_raw_len != float('inf'):
                        original_num_examples = len(raw_dataset)
                        raw_dataset = raw_dataset.filter(
                            lambda ex: isinstance(ex.get(text_col), str) and min_raw_len <= len(ex.get(text_col, "")) <= max_raw_len,
                            num_proc=4 
                        )
                        print(f"  Filtered raw dataset from {original_num_examples} to {len(raw_dataset)} based on raw char length [{min_raw_len}, {max_raw_len}].")

                    if len(raw_dataset) == 0:
                        print("  Raw dataset is empty after filtering by character length.")
                    
                    chunk_size = self.dataset_args.get("max_tokenized_length", 512)
                    stride = self.dataset_args.get("stride_for_chunking", chunk_size // 2)
                    min_chunk_len = self.dataset_args.get("min_tokenized_length", 64)

                    for example in raw_dataset:
                        text = example.get(text_col)
                        if not text or not isinstance(text, str): continue
                        text = " ".join(text.split()) 
                        tokenized_document = self.tokenizer(text, add_special_tokens=False).input_ids
                        if not tokenized_document: continue
                        for i in range(0, len(tokenized_document) - min_chunk_len + 1, stride):
                            chunk = tokenized_document[i : i + chunk_size]
                            if len(chunk) >= min_chunk_len:
                               self.all_tokenized_chunks.append(chunk)
                    
                    if not self.all_tokenized_chunks:
                        print(f"Warning: No valid chunks generated from dataset {dataset_name}.")
                    else:
                        random.shuffle(self.all_tokenized_chunks)
                        print(f"Dataset processed into {len(self.all_tokenized_chunks)} tokenized chunks.")
                except Exception as e:
                    print(f"Error loading or processing HF dataset '{dataset_name}': {e}. Will use default prompt.")
        
        elif source_type == "fixed_list":
            fixed_texts = self.dataset_args.get("fixed_texts", [])
            if not fixed_texts: print("Warning: source_type is 'fixed_list' but no 'fixed_texts' provided.")
            for text in fixed_texts:
                token_ids = self.tokenizer(text, truncation=True, 
                                           max_length=self.dataset_args.get("max_tokenized_length", 512)).input_ids
                if len(token_ids) >= self.dataset_args.get("min_tokenized_length", 10):
                    self.all_tokenized_chunks.append(token_ids)
            if self.all_tokenized_chunks:
                self.all_tokenized_chunks = self.all_tokenized_chunks * self.dataset_args.get("fixed_list_repeat", 50)
                random.shuffle(self.all_tokenized_chunks)
                print(f"Using fixed list, repeated to {len(self.all_tokenized_chunks)} chunks.")
        
        if not self.all_tokenized_chunks:
             self.all_tokenized_chunks.append(
                 self.tokenizer("This is a default fallback segment for testing.", return_tensors="pt").input_ids[0].tolist()
             )
             print("Initialized with a single default fallback segment due to data loading issues.")
        self.dataset_chunk_iterator = iter(self.all_tokenized_chunks)

    def _get_next_text_segment_from_dataset(self) -> List[int]: # (与您之前能成功加载数据的版本一致)
        try:
            return next(self.dataset_chunk_iterator)
        except StopIteration:
            if not self.all_tokenized_chunks: 
                return self.tokenizer("Fallback segment, list was empty.", return_tensors="pt").input_ids[0].tolist()
            random.shuffle(self.all_tokenized_chunks) 
            self.dataset_chunk_iterator = iter(self.all_tokenized_chunks)
            try: return next(self.dataset_chunk_iterator)
            except StopIteration: return self.all_tokenized_chunks[0] 
        except Exception as e:
            print(f"Error getting next text segment: {e}. Returning default.")
            return self.tokenizer("Error fallback segment.", return_tensors="pt").input_ids[0].tolist()

    def reset(self) -> None: # (与您之前能成功加载数据的版本一致)
        self.comp_cache._initialize_cache_lists() 
        self._past_key_values_ref = None 
        self.ema_logits_ref = None 
        self._current_text_segment_token_ids = self._get_next_text_segment_from_dataset()
        self._current_segment_processed_tokens = 0 
        self._current_episode_total_tokens_generated = 0 

        prompt_len_initial = min(len(self._current_text_segment_token_ids) // 2, 
                                 self.llm_config_params.get("max_context_length_for_llm_input", 1024) // 2) #
        prompt_len_initial = max(1, prompt_len_initial) 

        initial_input_ids_list = self._current_text_segment_token_ids[:prompt_len_initial]
        if not initial_input_ids_list: # 如果切片后为空
            initial_input_ids_list = [self.tokenizer.bos_token_id or 0] # 使用BOS作为安全默认值
            print("Warning: Initial prompt slice was empty, using BOS token.")

        initial_input_ids = torch.tensor([initial_input_ids_list], dtype=torch.long, device=self.device)
        self._current_segment_processed_tokens = initial_input_ids.shape[1] # 更新为实际使用的prompt长度
        
        if initial_input_ids.shape[1] == 0: 
            initial_input_ids = torch.tensor([[self.tokenizer.bos_token_id or 0]], dtype=torch.long, device=self.device)
            self._current_segment_processed_tokens = 1
            if not self._current_text_segment_token_ids:
                 self._current_text_segment_token_ids = initial_input_ids[0].tolist()

        attention_mask = torch.ones_like(initial_input_ids)
        with torch.no_grad():
            outputs_main_init = self.llm_model( # Renamed to avoid confusion
                input_ids=initial_input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True, return_dict=True
            )
            if outputs_main_init.past_key_values:
                self.comp_cache.update_from_hf_past(outputs_main_init.past_key_values)
            
            outputs_ref_init = self.llm_model( # Renamed to avoid confusion
                input_ids=initial_input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True, return_dict=True
            )
            self._past_key_values_ref = outputs_ref_init.past_key_values
            if outputs_ref_init.logits is not None and outputs_ref_init.logits.numel() > 0:
                initial_logits_for_ema = outputs_ref_init.logits[:, -1, :].detach().clone()
                self.ema_logits_ref = initial_logits_for_ema
            else:
                self.ema_logits_ref = None

    def _calculate_loss(self, logits_compressed: torch.Tensor, logits_target: torch.Tensor) -> torch.Tensor: # logits_target可以是ema_logits_ref
        loss_type = self.training_params.get("LOSS_FUNCTION", "TOP_K_KL").upper()
        top_k = self.training_params.get("TOP_K_LOGITS", 10) 
        temperature = self.training_params.get("KL_TEMPERATURE", 1.0)
        logits_target_detached = logits_target.detach() # 确保目标不参与梯度

        if "TOP_K" in loss_type:
            probs_target_full = F.softmax(logits_target_detached / temperature, dim=-1)
            top_k_probs_target_values, top_k_indices_target = torch.topk(probs_target_full, k=top_k, dim=-1, sorted=True)
            
            # 使用 gather 从 logits_compressed (而不是其softmax) 中获取对应位置的值，以便后续计算更灵活
            # logits_compressed_selected_for_top_k shape: (batch_size, top_k)
            batch_idx = torch.arange(logits_compressed.shape[0], device=logits_compressed.device).unsqueeze(1)
            logits_compressed_selected = logits_compressed[batch_idx, top_k_indices_target]

            # 如果使用KL，需要概率；如果使用MSE，可以在logits上或概率上
            if loss_type == "TOP_K_KL":
                probs_compressed_selected = F.softmax(logits_compressed_selected / temperature, dim=-1)
                # 归一化选出的Top-K概率 (因为它们是从不同分布中选出的子集)
                epsilon_norm = 1e-9
                norm_top_k_probs_target = top_k_probs_target_values / (torch.sum(top_k_probs_target_values, dim=-1, keepdim=True) + epsilon_norm)
                norm_probs_compressed_selected = probs_compressed_selected / (torch.sum(probs_compressed_selected, dim=-1, keepdim=True) + epsilon_norm)
                
                log_norm_probs_compressed_selected = torch.log(norm_probs_compressed_selected + epsilon_norm)
                loss = F.kl_div(log_norm_probs_compressed_selected, norm_top_k_probs_target, 
                                reduction='batchmean', log_target=False)
            elif loss_type == "TOP_K_MSE":
                # 在选出的Top-K logits上计算MSE (通常更稳定)
                logits_target_selected = logits_target_detached[batch_idx, top_k_indices_target]
                loss = F.mse_loss(logits_compressed_selected, logits_target_selected)
            else:
                raise ValueError(f"Unsupported TOP_K loss type: {loss_type}")
        elif loss_type == "KL": # 全分布KL
            log_probs_compressed = F.log_softmax(logits_compressed / temperature, dim=-1)
            probs_ref = F.softmax(logits_target_detached / temperature, dim=-1)
            loss = F.kl_div(log_probs_compressed, probs_ref, reduction='batchmean', log_target=False)
        elif loss_type == "MSE": # 全分布MSE
            loss = F.mse_loss(logits_compressed, logits_target_detached)
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
        if self._current_segment_processed_tokens >= len(self._current_text_segment_token_ids):
            self.reset()
            if self._current_segment_processed_tokens >= len(self._current_text_segment_token_ids):
                # print("DEBUG ENV: Reset called, but segment still exhausted or empty.")
                return torch.tensor(0.0, device=self.device, requires_grad=False), True, {"error": "Empty segment after reset"}

        current_input_token_id = self._current_text_segment_token_ids[self._current_segment_processed_tokens]
        current_input_ids = torch.tensor([[current_input_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(current_input_ids)

        # --- 1. 参考LLM路径 (获取 logits_ref 和更新参考KV缓存及EMA) ---
        logits_ref: Optional[torch.Tensor] = None
        newly_generated_token_ref: Optional[int] = -1 
        try:
            with torch.no_grad(): # 参考路径不计算梯度
                self.llm_model.eval()
                outputs_ref = self.llm_model(
                    input_ids=current_input_ids, attention_mask=attention_mask,
                    past_key_values=self._past_key_values_ref, use_cache=True, return_dict=True
                )
                if outputs_ref.logits is not None:
                    logits_ref = outputs_ref.logits[:, -1, :]
                    self._past_key_values_ref = outputs_ref.past_key_values 
                    current_logits_ref_detached = logits_ref.detach().clone() # 为EMA复制并分离
                    if self.ema_logits_ref is None: self.ema_logits_ref = current_logits_ref_detached
                    else: self.ema_logits_ref = self.ema_alpha * current_logits_ref_detached + (1.0 - self.ema_alpha) * self.ema_logits_ref
                
                next_token_id_ref_tensor = torch.argmax(logits_ref, dim=-1) if logits_ref is not None else torch.tensor(-1)
                newly_generated_token_ref = next_token_id_ref_tensor.item()
        except Exception as e:
            print(f"Error during Reference LLM forward: {e}")
            return None, True, {"error_ref": str(e), "stage": "ref_llm"}
        
        if logits_ref is None or self.ema_logits_ref is None:
            print("Error: Reference logits or EMA reference logits are None after ref LLM step.")
            return torch.tensor(self.training_params.get("ERROR_LOSS_VALUE", 10.0), device=self.device, requires_grad=True), True, {"error": "Ref logits None"}

        # --- 主LLM路径与压缩 ---
        logits_compressed: Optional[torch.Tensor] = None
        newly_generated_token_main: Optional[int] = -1
        compression_performed_this_step = False
        
        try:
            # 整个主路径操作都在梯度追踪下
            with torch.set_grad_enabled(True): 
                self.llm_model.eval() # LLM参数固定
                _ = self.llm_model( 
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache, # 传入当前的CompCache实例
                    use_cache=True, return_dict=True 
                )

                compression_triggered_this_step = False
                for layer_idx_check in self.comp_cache.config.compress_layer_ids:
                    if layer_idx_check < len(self.comp_cache.key_cache) and \
                       self.comp_cache.key_cache[layer_idx_check].ndim == 4: # 确保层已初始化为4D
                        layer_total_len = self.comp_cache.key_cache[layer_idx_check].shape[-2]
                        if (layer_total_len - self.initial_keep_len) >= self.compression_trigger_threshold:
                            compression_triggered_this_step = True; break
                
                if compression_triggered_this_step:
                     compression_performed_this_step = self.comp_cache.trigger_global_compression()

                outputs_main_final = self.llm_model(
                    input_ids=current_input_ids, 
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache, # 使用已更新和（可能）压缩的缓存
                    use_cache=True, # 再次调用CompCache.update (确保与推理行为一致)
                    return_dict=True
                )
                if outputs_main_final.logits is not None:
                    logits_compressed = outputs_main_final.logits[:, -1, :]
                    # print(f"DEBUG ENV: logits_compressed (final) grad_fn: {logits_compressed.grad_fn}")
                
                if logits_compressed is not None:
                    next_token_id_main_tensor = torch.argmax(logits_compressed.detach(), dim=-1)
                    newly_generated_token_main = next_token_id_main_tensor.item()

        except Exception as e:
            print(f"Error during Main LLM processing or compression in generate_one_step: {e}")
            import traceback; traceback.print_exc()
            return None, True, {"error_main_llm_path": str(e)}

        # --- 更新处理token的计数器 ---
        self._current_segment_processed_tokens += 1
        self._current_episode_total_tokens_generated += 1

        # --- 计算损失 ---
        loss = torch.tensor(0.0, device=self.device, requires_grad=False) 
        if logits_compressed is not None and self.ema_logits_ref is not None:
            loss = self._calculate_loss(logits_compressed, self.ema_logits_ref) 
            if loss.grad_fn is None and loss.requires_grad: # 通常loss是计算的结果，它应该有grad_fn
                 print(f"WARNING: Calculated loss has no grad_fn but requires_grad is True. Loss: {loss.item()}")
            # elif loss.requires_grad:
                 # print(f"DEBUG ENV: Calculated Loss grad_fn: {loss.grad_fn}")

        elif logits_compressed is None:
            print("Warning: logits_compressed is None for loss calculation.")
            loss = torch.tensor(self.training_params.get("ERROR_LOSS_VALUE", 10.0), device=self.device, requires_grad=True) 
        elif self.ema_logits_ref is None: # logits_compressed is not None but ema_logits_ref is
            print("Warning: ema_logits_ref is None for loss calculation (logits_compressed is valid).")
            loss = torch.tensor(self.training_params.get("ERROR_LOSS_VALUE", 10.0), device=self.device, requires_grad=True)
        
        done = self._current_episode_total_tokens_generated >= self.training_params["MAX_TOKENS_PER_EPISODE"] or \
               self._current_segment_processed_tokens >= len(self._current_text_segment_token_ids)
        
        info = {
            "loss_value": loss.item() if loss is not None else float('nan'),
            "compressed_logits_entropy": self._calculate_logits_entropy(logits_compressed),
            "ref_logits_entropy": self._calculate_logits_entropy(logits_ref), 
            "compression_triggered_this_step": compression_triggered_this_step,
            "compression_performed_this_step": compression_performed_this_step,
            "predicted_token_main": newly_generated_token_main if newly_generated_token_main is not None else -1,
            "predicted_token_ref": newly_generated_token_ref if newly_generated_token_ref is not None else -1,
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