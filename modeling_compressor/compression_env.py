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
                 training_params: Dict[str, Any] = TRAINING_PARAMS, # training_params 现在包含 curriculum_config 和 dataset_args
                 llm_config_params: Dict[str, Any] = LLM_CONFIG,
                 current_stage_dataset_args: Optional[Dict[str, Any]] = None): 
        print("Initializing Compression Environment with Dataset Support...")
        self.training_params = training_params
        self.llm_config_params = llm_config_params
        self.dataset_args = current_stage_dataset_args if current_stage_dataset_args is not None \
                            else training_params.get("dataset_args", {})
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

        # 动态调整CompressorConfig
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
        self.all_tokenized_chunks: List[List[int]] = [] # 存储所有预处理好的token块
        self.dataset_chunk_iterator: Optional[iter] = None
        self._load_and_prepare_dataset() # 加载并处理数据集

        self._current_text_segment_token_ids: List[int] = []
        self._current_segment_processed_tokens = 0
        self._past_key_values_ref = None
        self._current_episode_total_tokens_generated = 0
        self.compression_trigger_threshold = compressor_config.compression_threshold
        self.initial_keep_len = compressor_config.initial_uncompressed_keep_length
        print(f"Compression Environment initialized for dataset args: {self.dataset_args.get('hf_dataset_name', 'default prompt')}")

        self.compression_trigger_threshold = compressor_config.compression_threshold
        self.initial_keep_len = compressor_config.initial_uncompressed_keep_length
        print("Compression Environment initialized successfully with dataset.")
        self.ema_alpha = self.training_params.get("EMA_ALPHA_LOGITS_REF", 0.1) # 从配置获取alpha
        self.ema_logits_ref: Optional[torch.Tensor] = None # 初始化EMA平滑参考Logits
        print(f"Using EMA for reference logits with alpha: {self.ema_alpha}")

    def _load_and_prepare_dataset(self):
        """根据 self.dataset_args 加载并准备数据集，将其切分为token块。"""
        source_type = self.dataset_args.get("source_type", "huggingface_dataset")
        self.all_tokenized_chunks = [] # 清空

        if source_type == "huggingface_dataset":
            dataset_name = self.dataset_args.get("hf_dataset_name")
            if not dataset_name:
                print("No hf_dataset_name provided. Using default prompt for data generation.")
                return # 将依赖 _get_next_text_segment_from_dataset 中的回退逻辑

            print(f"Loading HF dataset: {dataset_name} with config: {self.dataset_args.get('hf_dataset_config_name')}")
            try:
                num_samples_str = str(self.dataset_args.get("max_samples_to_load", ""))
                split_arg = self.dataset_args.get("hf_split", "train")
                if num_samples_str: # e.g., "train[:1000]" or "train[:10%]"
                     if "%" in num_samples_str or ":" in split_arg : # 如果split_arg本身已经是切片形式
                         pass # datasets库能处理 train[:1000][:10%] 这样的情况，但最好避免
                     else: # 简单拼接
                         split_arg = f"{split_arg}[{num_samples_str}]" if ":" not in num_samples_str else f"{split_arg}{num_samples_str}"


                raw_dataset = load_dataset(
                    dataset_name,
                    self.dataset_args.get("hf_dataset_config_name"),
                    split=split_arg,
                    # streaming=True, # 流式加载对分块和打乱更复杂
                )
                
                # 过滤原始文本长度 (可选，但有助于获取同质化数据)
                min_raw_len = self.dataset_args.get("min_raw_text_length", 0)
                max_raw_len = self.dataset_args.get("max_raw_text_length", float('inf'))
                text_col = self.dataset_args["hf_text_column"]
                
                if min_raw_len > 0 or max_raw_len != float('inf'):
                    original_num_examples = len(raw_dataset)
                    raw_dataset = raw_dataset.filter(
                        lambda ex: isinstance(ex[text_col], str) and min_raw_len <= len(ex[text_col]) <= max_raw_len
                    ) # 添加 isinstance 检查
                    print(f"  Filtered raw dataset from {original_num_examples} to {len(raw_dataset)} based on raw char length [{min_raw_len}, {max_raw_len}].")

                if len(raw_dataset) == 0:
                    print("  Raw dataset is empty after filtering by character length. No chunks will be generated.")
            
                # 分块和Tokenize
                chunk_size = self.dataset_args.get("max_tokenized_length", 512)
                stride = self.dataset_args.get("stride_for_chunking", chunk_size // 2)
                min_chunk_len = self.dataset_args.get("min_tokenized_length", 64)

                for example in raw_dataset:
                    text = example[text_col]
                    if not text or not isinstance(text, str): continue
                    text = " ".join(text.split()) # 清理空格
                    
                    # 先对整个文档tokenize，避免在循环中重复tokenize相同前缀
                    # add_special_tokens=False 是为了避免在文档中间加入bos/eos，除非LLM期望这样
                    tokenized_document = self.tokenizer(text, add_special_tokens=False).input_ids
                    
                    if not tokenized_document: continue

                    for i in range(0, len(tokenized_document) - min_chunk_len + 1, stride):
                        chunk = tokenized_document[i : i + chunk_size]
                        if len(chunk) >= min_chunk_len: # 确保块至少有最小长度
                           self.all_tokenized_chunks.append(chunk)
                
                if not self.all_tokenized_chunks:
                    print(f"Warning: No valid chunks generated from dataset {dataset_name}. Check filters and chunking params.")
                else:
                    random.shuffle(self.all_tokenized_chunks) # 打乱所有块
                    print(f"Dataset processed into {len(self.all_tokenized_chunks)} tokenized chunks.")

            except Exception as e:
                print(f"Error loading or processing HF dataset '{dataset_name}': {e}. Will use default prompt.")
        
        elif source_type == "fixed_list": # 用于第一阶段的固定文本
            fixed_texts = self.dataset_args.get("fixed_texts", [])
            if not fixed_texts:
                print("Warning: source_type is 'fixed_list' but no 'fixed_texts' provided. Using default prompt.")
            for text in fixed_texts:
                token_ids = self.tokenizer(text, truncation=True, 
                                           max_length=self.dataset_args.get("max_tokenized_length", 512)).input_ids
                if len(token_ids) >= self.dataset_args.get("min_tokenized_length", 10):
                    self.all_tokenized_chunks.append(token_ids)
            if self.all_tokenized_chunks:
                # 对于固定列表，可以重复使用以增加稳定性
                self.all_tokenized_chunks = self.all_tokenized_chunks * self.dataset_args.get("fixed_list_repeat", 50)
                random.shuffle(self.all_tokenized_chunks)
                print(f"Using fixed list of {len(fixed_texts)} texts, repeated to {len(self.all_tokenized_chunks)} chunks.")
        else:
            print(f"Unsupported dataset_source_type: {source_type}. Using default prompt.")

        if not self.all_tokenized_chunks: # 如果没有任何数据块
             self.all_tokenized_chunks.append(
                 self.tokenizer("This is a default fallback segment.", return_tensors="pt").input_ids[0].tolist()
             )
             print("Initialized with a single default fallback segment.")
        
        self.dataset_chunk_iterator = iter(self.all_tokenized_chunks)


    def _get_next_text_segment_from_dataset(self) -> List[int]:
        """从预处理好的 all_tokenized_chunks 中获取下一个块。"""
        try:
            return next(self.dataset_chunk_iterator)
        except StopIteration:
            if not self.all_tokenized_chunks: # 如果列表本身是空的（不应发生，因为有fallback）
                return self.tokenizer("Fallback segment, list was empty.", return_tensors="pt").input_ids[0].tolist()
            
            # print("Chunk iterator exhausted. Re-shuffling all_tokenized_chunks.")
            random.shuffle(self.all_tokenized_chunks) # 重新打乱所有块
            self.dataset_chunk_iterator = iter(self.all_tokenized_chunks)
            try:
                return next(self.dataset_chunk_iterator)
            except StopIteration: # 如果 all_tokenized_chunks 仍然是空的（例如只有一个元素且已用完）
                 return self.all_tokenized_chunks[0] # 返回第一个（也是唯一一个）
        except Exception as e:
            print(f"Error getting next text segment: {e}. Returning default.")
            return self.tokenizer("Error fallback segment.", return_tensors="pt").input_ids[0].tolist()



    def _get_initial_prompt_ids(self, batch_size: int = 1) -> torch.Tensor: # (保持不变)
        initial_text = "User: Hello, let's discuss advanced AI topics.\nAgent:"
        inputs = self.tokenizer(initial_text, return_tensors="pt", padding=False, truncation=True, 
                                max_length=self.llm_config_params["max_context_length_for_llm_input"] // 4)
        return inputs.input_ids.to(self.device)

    def reset(self) -> None:
        """重置环境，加载新的文本段作为上下文"""
        self.comp_cache._initialize_cache_lists() # 清空并重置 CompCache
        self._past_key_values_ref = None # 重置参考路径的KV缓存
        self.ema_logits_ref = None # 重置EMA平滑参考Logits
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
            # 用预填充后的第一个有效 logits_ref 初始化 ema_logits_ref
            if outputs_ref.logits is not None and outputs_ref.logits.numel() > 0:
                initial_logits_for_ema = outputs_ref.logits[:, -1, :].detach().clone() # 取最后一个token的logits
                self.ema_logits_ref = initial_logits_for_ema
            else:
                self.ema_logits_ref = None # 如果预填充没有产生有效logits
        
    def _calculate_loss(self, logits_compressed: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor:
        """
        计算压缩logits与参考logits之间的损失，基于Top-K匹配。
        """
        loss_type = self.training_params.get("LOSS_FUNCTION", "TOP_K_KL").upper()
        top_k = self.training_params.get("TOP_K_LOGITS", 10) # 从配置获取K值
        temperature = self.training_params.get("KL_TEMPERATURE", 1.0) # 温度参数

        # 确保参考logits不参与梯度计算
        logits_ref_detached = logits_ref.detach()

        # 1. 对参考logits进行softmax并获取Top-K
        probs_ref_full = F.softmax(logits_ref_detached / temperature, dim=-1)
        top_k_probs_ref_values, top_k_indices_ref = torch.topk(probs_ref_full, k=top_k, dim=-1, sorted=True)
        # 2. 对压缩logits进行softmax
        probs_compressed_full = F.softmax(logits_compressed / temperature, dim=-1)

        # 3. 从压缩logits的概率分布中，根据参考logits的Top-K索引，提取对应的概率值
        batch_size = probs_compressed_full.shape[0]
        if batch_size == 1: # 常见情况
            probs_compressed_selected_for_top_k_ref = probs_compressed_full[0, top_k_indices_ref[0]]
            probs_compressed_selected_for_top_k_ref = probs_compressed_selected_for_top_k_ref.unsqueeze(0) # 恢复batch维度
        else: # 处理 batch_size > 1 的情况
            probs_compressed_selected_for_top_k_ref = torch.gather(
                probs_compressed_full, 
                dim=1, # 在 vocab_size 维度上 gather
                index=top_k_indices_ref
            )
        # 4. 计算损失
        if loss_type == "TOP_K_KL":
            epsilon_norm = 1e-9
            norm_top_k_probs_ref = top_k_probs_ref_values / (torch.sum(top_k_probs_ref_values, dim=-1, keepdim=True) + epsilon_norm)
            norm_probs_compressed_selected = probs_compressed_selected_for_top_k_ref / (torch.sum(probs_compressed_selected_for_top_k_ref, dim=-1, keepdim=True) + epsilon_norm)
            
            log_norm_probs_compressed_selected = torch.log(norm_probs_compressed_selected + epsilon_norm)
            
            loss = F.kl_div(log_norm_probs_compressed_selected, norm_top_k_probs_ref, 
                            reduction='batchmean', log_target=False) # log_target=False因为norm_top_k_probs_ref是概率

        elif loss_type == "TOP_K_MSE":
            # 直接在选出的Top-K概率上计算MSE
            loss = F.mse_loss(probs_compressed_selected_for_top_k_ref, top_k_probs_ref_values)
        
        elif loss_type == "TOP_K_CROSS_ENTROPY": # 这种方式可能更常用
            print(f"Warning: TOP_K_CROSS_ENTROPY not fully implemented in this simplified Top-K. Falling back to TOP_K_MSE.")
            loss = F.mse_loss(probs_compressed_selected_for_top_k_ref, top_k_probs_ref_values)

        else: # 如果LOSS_FUNCTION不是上面明确处理的Top-K类型，但包含了TOP_K字样，则报错或默认
            if "TOP_K" in loss_type:
                print(f"Warning: Unsupported TOP_K loss type: {loss_type}. Defaulting to TOP_K_MSE.")
                loss = F.mse_loss(probs_compressed_selected_for_top_k_ref, top_k_probs_ref_values)
            else: # 如果是原始的 "KL" 或 "MSE" (整个分布)
                raise ValueError(f"Loss function {loss_type} called in Top-K loss calculation. Ensure LOSS_FUNCTION in config is a TOP_K type.")
        
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
                return torch.tensor(0.0, device=self.device, requires_grad=False), True, {"error": "Empty segment after reset"}

        current_input_token_id = self._current_text_segment_token_ids[self._current_segment_processed_tokens]
        current_input_ids = torch.tensor([[current_input_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(current_input_ids)

        # --- 1. 参考LLM路径 (获取 logits_ref 和更新参考KV缓存) ---
        logits_ref: Optional[torch.Tensor] = None
        newly_generated_token_ref: Optional[int] = None # 用于日志
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
                if logits_ref is not None: # 更新EMA也在这里
                    current_logits_ref_detached = logits_ref.detach().clone()
                    if self.ema_logits_ref is None:
                        self.ema_logits_ref = current_logits_ref_detached
                    else:
                        self.ema_logits_ref = self.ema_alpha * current_logits_ref_detached + \
                                             (1.0 - self.ema_alpha) * self.ema_logits_ref
            next_token_id_ref_tensor = torch.argmax(logits_ref.detach(), dim=-1) if logits_ref is not None else torch.tensor(-1)
            # newly_generated_token_ref = next_token_id_ref_tensor.item()
        except Exception as e:
            print(f"Error during Reference LLM forward: {e}")
            return None, True, {"error_ref": str(e)}

        # --- 2. 主LLM路径 ---
        # 确保K和V压缩器参数在进行任何操作前都是可训练的（如果之前被冻结过）
        # 这个操作在 train_script.py 的循环开始处已经做了，这里可以省略，
        # 因为 generate_one_step 不应该负责这个全局状态。
        # set_requires_grad(self.comp_cache.k_compressor, True)
        # set_requires_grad(self.comp_cache.v_compressor, True)

        logits_compressed: Optional[torch.Tensor] = None
        newly_generated_token_main: Optional[int] = None
        
        # 2.A. 让主LLM（带CompCache）的forward隐式调用CompCache.update，
        #      以将当前 current_input_ids 对应的（未压缩）KV添加到CompCache中。
        #      这次调用不直接用于loss的logits，主要目的是更新CompCache的内部状态。
        #      为了避免干扰后续的梯度计算，这次可以放在no_grad下，或者我们接受它构建的图。
        #      如果放在no_grad下，那么CompCache.update中的key_states就没有梯度信息。
        #      更优的做法是让它在grad_enabled下进行，但我们不使用它的logits输出。
        try:
            with torch.set_grad_enabled(True): # 保持梯度追踪，因为update会修改参与后续计算的缓存
                self.llm_model.eval()
                # 确保压缩器参数是可训练的，以便后续压缩能构建图
                for param in self.comp_cache.k_compressor.parameters(): param.requires_grad = True
                for param in self.comp_cache.v_compressor.parameters(): param.requires_grad = True

                # 第一次forward，主要是为了CompCache.update被调用
                # 其输出的past_key_values就是更新后的self.comp_cache
                _ = self.llm_model( 
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache, # 传入当前的CompCache
                    use_cache=True, return_dict=True 
                )
                # 此调用后，self.comp_cache (通过其update方法) 已包含current_input_ids的KV
        except Exception as e:
            print(f"Error during Main LLM pre-update forward: {e}")
            return None, True, {"error_main_pre_update": str(e)}

        # 2.B. 根据固定规则和更新后的CompCache状态，决定是否触发压缩
        compression_triggered_this_step = False
        for layer_idx_check in self.comp_cache.config.compress_layer_ids:
            if layer_idx_check < len(self.comp_cache.key_cache) and \
               self.comp_cache.key_cache[layer_idx_check].ndim == 4:
                layer_total_len = self.comp_cache.key_cache[layer_idx_check].shape[-2]
                if (layer_total_len - self.initial_keep_len) >= self.compression_trigger_threshold:
                    compression_triggered_this_step = True
                    break
        
        compression_performed_this_step = False
        if compression_triggered_this_step:
            with torch.set_grad_enabled(True): # 确保压缩操作在梯度追踪下
                 # KVCompressor参数的requires_grad已在上面或外部设为True
                 compression_performed_this_step = self.comp_cache.trigger_global_compression()
                 # trigger_global_compression 调用 KVCompressor.forward()
                 # 并用其输出更新 self.comp_cache.key_cache 和 value_cache
                 # 这些更新后的缓存张量现在连接到了KVCompressor的参数
        
        # 2.C. 主LLM再次（或主要的一次）forward，以获取受当前步骤压缩决策影响的 logits_compressed
        try:
            with torch.set_grad_enabled(True): 
                self.llm_model.eval()
                # KVCompressor参数的requires_grad应已正确设置
                outputs_main_final = self.llm_model(
                    input_ids=current_input_ids, # 仍然是本步骤的输入token
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache, # 使用当前（在update和可能压缩之后）的CompCache
                    use_cache=True, # 让CompCache再次被update (这次主要是为了获取logits)
                    return_dict=True
                )
                logits_compressed = outputs_main_final.logits[:, -1, :]
            
            next_token_id_main_tensor = torch.argmax(logits_compressed.detach(), dim=-1) if logits_compressed is not None else torch.tensor(-1)
            newly_generated_token_main = next_token_id_main_tensor.item()
        except Exception as e:
            print(f"Error during Main LLM final forward: {e}")
            return None, True, {"error_main_final_forward": str(e)}

        # --- 更新对话历史和计数器 ---
        if newly_generated_token_main is not None and newly_generated_token_main != -1:
            pass # 不再需要追加到 _current_dialogue_token_ids 或 _current_text_segment_token_ids
        self._current_segment_processed_tokens += 1
        self._current_episode_total_tokens_generated += 1

        # --- 计算损失 ---
        loss = torch.tensor(0.0, device=self.device, requires_grad=False) # 默认无梯度
        if logits_compressed is not None and self.ema_logits_ref is not None:
            # _calculate_loss的输出应该是一个需要梯度的标量
            calculated_loss = self._calculate_loss(logits_compressed, self.ema_logits_ref)
            loss = calculated_loss 
        elif logits_compressed is not None and logits_ref is not None and self.ema_logits_ref is None:
            print("Warning: ema_logits_ref is None, falling back to instantaneous logits_ref for loss.")
            loss = self._calculate_loss(logits_compressed, logits_ref)
        else:
            print("Warning: Logits missing for loss calculation.")
            # 如果希望出错时有惩罚或可反向传播的0，需要 loss = torch.tensor(VALUE, device=self.device, requires_grad=True)
            # 但如果是因为错误，可能不应该反向传播。
            # 对于不收敛，如果这里返回的loss不带梯度，那参数永远不会更新。
            # 确保_calculate_loss返回的张量有grad_fn

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