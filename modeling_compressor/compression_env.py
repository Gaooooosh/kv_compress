# compression_env.py (Modified for Simplified Global CompCache)
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import random

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
        print("Initializing Compression Environment (for Global Multi-Layer Compression)...")
        self.training_params = training_params
        self.llm_config_params = llm_config_params
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
        llm_num_layers = self.llm_model.config.num_hidden_layers
        llm_num_heads = self.llm_model.config.num_attention_heads
        llm_num_kv_heads = getattr(self.llm_model.config, 'num_key_value_heads', llm_num_heads)
        llm_hidden_size = self.llm_model.config.hidden_size
        llm_head_dim = llm_hidden_size // llm_num_heads
        expected_compressor_input_dim = llm_num_kv_heads * llm_head_dim

        # 更新传入的 compressor_config 实例的属性
        # 这是因为KVCompressor实例化时需要这些准确的值
        # 而CompCache本身也需要正确的layer_nums来初始化其列表
        original_comp_layer_ids = compressor_config.compress_layer_ids # 保存原始的，以防被覆盖
        
        compressor_config.layer_nums = llm_num_layers # CompCache内部列表长度基于此
        if compressor_config.input_dim != expected_compressor_input_dim:
            compressor_config.input_dim = expected_compressor_input_dim
        if compressor_config.kv_head_dim != llm_num_kv_heads:
            compressor_config.kv_head_dim = llm_num_kv_heads
        
        # 确保 compressor_config.compress_layer_ids 是有效的
        # （例如，如果用户配置跳过前4层，但模型总共只有2层，就需要调整）
        valid_compress_layer_ids = []
        if original_comp_layer_ids: # 如果用户提供了
            for layer_id in original_comp_layer_ids:
                if 0 <= layer_id < llm_num_layers:
                    valid_compress_layer_ids.append(layer_id)
            if not valid_compress_layer_ids and llm_num_layers > 0 : # 如果用户提供的都无效，但模型有层
                 # print(f"Warning: Original compress_layer_ids {original_comp_layer_ids} are invalid for {llm_num_layers} layers. Defaulting to compressing last layer if possible.")
                 # valid_compress_layer_ids = [llm_num_layers -1] # 默认压缩最后一层
                 pass # 或者让它为空，意味着不压缩任何特定层，由全局逻辑决定
        compressor_config.compress_layer_ids = valid_compress_layer_ids # 更新配置
        # print(f"Adjusted CompressorConfig: layer_nums={compressor_config.layer_nums}, input_dim={compressor_config.input_dim}, kv_head_dim={compressor_config.kv_head_dim}, compress_ids={compressor_config.compress_layer_ids}")

        self.comp_cache = CompCache(config=compressor_config, device=self.device)
        print("CompCache initialized with (potentially adjusted) CompressorConfig.")

        self._current_dialogue_token_ids: List[int] = []
        self._past_key_values_ref: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None
        self._current_episode_total_tokens = 0
        
        # 固定压缩规则参数 (从CompressorConfig读取)
        self.compression_trigger_threshold = compressor_config.compression_threshold
        self.initial_keep_len = compressor_config.initial_uncompressed_keep_length
        # print(f"Compression will be triggered if compressible zone in any specified layer >= {self.compression_trigger_threshold}")

        print("Compression Environment initialized successfully.")

    def _get_initial_prompt_ids(self, batch_size: int = 1) -> torch.Tensor: # (保持不变)
        initial_text = "User: Hello, let's discuss advanced AI topics.\nAgent:"
        inputs = self.tokenizer(initial_text, return_tensors="pt", padding=False, truncation=True, 
                                max_length=self.llm_config_params["max_context_length_for_llm_input"] // 4)
        return inputs.input_ids.to(self.device)

    def reset(self) -> None: # (保持不变)
        self.comp_cache._initialize_cache_lists() # 使用CompCache内部的重置方法
        
        initial_input_ids = self._get_initial_prompt_ids()
        self._current_dialogue_token_ids = initial_input_ids[0].tolist()
        self._current_episode_total_tokens = initial_input_ids.shape[1]
        attention_mask = torch.ones_like(initial_input_ids)

        with torch.no_grad():
            outputs_main = self.llm_model(
                input_ids=initial_input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True, return_dict=True
            )
            if outputs_main.past_key_values:
                # 使用update_from_hf_past来用LLM处理prompt后的KV缓存填充CompCache
                # 这会正确设置CompCache的初始状态和_seen_tokens
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
        """
        环境执行一步生成，计算并返回用于训练KVCompressor的损失。
        """
        if not self._current_dialogue_token_ids:
            self.reset()

        last_token_id = self._current_dialogue_token_ids[-1]
        current_input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(current_input_ids)

        # --- 主LLM (使用CompCache) ---
        logits_compressed: Optional[torch.Tensor] = None
        newly_generated_token_main: Optional[int] = None
        
        # 直接传递CompCache实例
        # LLM的forward方法内部会调用CompCache.update来追加新生成的KV
        try:
            with torch.set_grad_enabled(True): 
                self.llm_model.eval()
                for param in self.comp_cache.k_compressor.parameters(): param.requires_grad = True
                for param in self.comp_cache.v_compressor.parameters(): param.requires_grad = True

                outputs_main = self.llm_model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self.comp_cache, # 直接传递CompCache实例
                    use_cache=True, 
                    return_dict=True
                )
                logits_compressed = outputs_main.logits[:, -1, :]
            
            next_token_id_main_tensor = torch.argmax(logits_compressed.detach(), dim=-1)
            newly_generated_token_main = next_token_id_main_tensor.item()
        except Exception as e:
            print(f"Error during Main LLM forward: {e}")
            import traceback; traceback.print_exc()
            return None, True, {} # 出错则回合结束


        # --- 参考LLM (无压缩) ---
        logits_ref: Optional[torch.Tensor] = None
        newly_generated_token_ref: Optional[int] = None
        try:
            with torch.no_grad():
                self.llm_model.eval()
                outputs_ref = self.llm_model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self._past_key_values_ref,
                    use_cache=True,
                    return_dict=True
                )
                logits_ref = outputs_ref.logits[:, -1, :]
                self._past_key_values_ref = outputs_ref.past_key_values
            
            next_token_id_ref_tensor = torch.argmax(logits_ref.detach(), dim=-1)
            newly_generated_token_ref = next_token_id_ref_tensor.item()
        except Exception as e:
            print(f"Error during Reference LLM forward: {e}")
            return None, True, {}


        # 更新对话历史 (使用主模型的生成token)
        if newly_generated_token_main is not None:
            self._current_dialogue_token_ids.append(newly_generated_token_main)
            self._current_dialogue_token_ids = self._current_dialogue_token_ids[-self.llm_config_params["max_context_length_for_llm_input"]:]
            self._current_episode_total_tokens += 1
        else:
            print("Warning: Main LLM did not generate new tokens in this step.")
            return torch.tensor(0.0, device=self.device, requires_grad=True), True, {"error": "Main LLM failed to generate"}


        # --- 固定规则触发全局压缩 ---
        # 规则：检查 self.comp_cache.config.compress_layer_ids 中的任何一层，
        #       其 current_total_len - initial_keep_len 是否达到 threshold
        compression_triggered_this_step = False
        compression_performed_this_step = False

        for layer_idx_check in self.comp_cache.config.compress_layer_ids:
            if layer_idx_check < len(self.comp_cache.key_cache) and \
               self.comp_cache.key_cache[layer_idx_check].ndim == 4 and \
               self.comp_cache.key_cache[layer_idx_check].numel() > 0:
                
                layer_total_len = self.comp_cache.key_cache[layer_idx_check].shape[-2]
                if (layer_total_len - self.initial_keep_len) >= self.compression_trigger_threshold:
                    compression_triggered_this_step = True
                    break 
        
        if compression_triggered_this_step:
            # print(f"  Step {self._current_episode_total_tokens}: Triggering global compression.")
            # 梯度追踪应在 trigger_global_compression 内部的压缩器调用时自动处理
            compression_performed_this_step = self.comp_cache.trigger_global_compression()


        # --- 计算损失 ---
        loss = torch.tensor(0.0, device=self.device) # 默认损失为0
        if logits_compressed is not None and logits_ref is not None:
            loss = self._calculate_loss(logits_compressed, logits_ref)
        else:
            print("Warning: Logits missing, loss set to 0 for this step.")
            # 如果希望出错时有惩罚，可以设一个较大的正值，但要确保可反向传播
            # loss = torch.tensor(10.0, device=self.device, requires_grad=True) 

        done = self._current_episode_total_tokens >= self.training_params["MAX_TOKENS_PER_EPISODE"]
        
        info = {
            "loss_value": loss.item(),
            "compressed_logits_entropy": self._calculate_logits_entropy(logits_compressed),
            "ref_logits_entropy": self._calculate_logits_entropy(logits_ref),
            "compression_triggered_this_step": compression_triggered_this_step,
            "compression_performed_this_step": compression_performed_this_step,
            "newly_generated_token_main": newly_generated_token_main if newly_generated_token_main is not None else -1,
            "newly_generated_token_ref": newly_generated_token_ref if newly_generated_token_ref is not None else -1,
            # "current_total_tokens_in_episode": self._current_episode_total_tokens # 用于外部调试
        }
        # 可以在info中添加CompCache的统计数据（如果需要监控）
        # layer0_stats = self._get_simplified_cache_stats(0) # 获取简化统计
        # info.update({f"L0_{k}": v for k,v in layer0_stats.items()})

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

if __name__ == '__main__':
    print("Testing Compression Environment (Simplified Global KVCompressor Training)...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # (测试脚本的 CompressorConfig 初始化部分需要与 CompCache 匹配，特别是 compress_layer_ids)
    try:
        from config_rl import DEFAULT_COMPRESSOR_CONFIG_PARAMS as DCCP, LLM_CONFIG as LCONF
        use_config_file_env_test = True
    except ImportError:
        DCCP = {
            "reduction_factor": 2, "output_seq_len": 4, "num_attention_heads": 2,
            "use_mixed_precision": False, "torch_dtype": "torch.float32",
            "kernel_size": 3, "padding": 1, "compression_threshold": 8, 
            "initial_uncompressed_keep_length": 2, "layer_nums": 4, 
            "kv_head_dim": 2, "input_dim": 128
        }
        # LCONF = {"model_name_or_path": "gpt2"} # gpt2作为默认的小模型
        use_config_file_env_test = False
        print("Using hardcoded DCCP and LCONF for env test.")

    test_layer_nums_env = DCCP.get("layer_nums", 4)
    test_kv_heads_env = DCCP.get("kv_head_dim", 2)
    test_input_dim_env = DCCP.get("input_dim", 128)

    if use_config_file_env_test and LCONF.get("model_name_or_path"):
        try:
            from transformers import AutoConfig
            llm_c = AutoConfig.from_pretrained(LCONF["model_name_or_path"])
            test_layer_nums_env = llm_c.num_hidden_layers
            test_kv_heads_env = getattr(llm_c, 'num_key_value_heads', llm_c.num_attention_heads)
            test_input_dim_env = (llm_c.hidden_size // llm_c.num_attention_heads) * test_kv_heads_env
        except Exception: pass # 忽略错误，使用默认值

    # 根据您的描述，例如跳过前4层
    skip_first_n_layers = 4 
    if test_layer_nums_env <= skip_first_n_layers : # 如果总层数不多，则调整
        if test_layer_nums_env > 1: # 至少压缩一层（如果有多于1层）
            compress_ids_test = list(range(1, test_layer_nums_env)) # 跳过第0层
        elif test_layer_nums_env == 1:
             compress_ids_test = [0] # 只有一层，就压缩它
        else: # 0层
            compress_ids_test = []
    else:
        compress_ids_test = list(range(skip_first_n_layers, test_layer_nums_env))
    
    print(f"Env Test: Total LLM Layers={test_layer_nums_env}, Layers to Compress IDs={compress_ids_test}")

    test_env_compressor_config = CompressorConfig(
        input_dim=test_input_dim_env,
        reduction_factor=DCCP.get("reduction_factor", 2),
        output_seq_len=DCCP.get("output_seq_len", 4),
        num_attention_heads=DCCP.get("num_attention_heads", 2),
        use_mixed_precision=False, torch_dtype=torch.float32,
        kv_head_dim=test_kv_heads_env,
        layer_nums=test_layer_nums_env, # 这是LLM的总层数
        kernel_size=DCCP.get("kernel_size", 3), padding=DCCP.get("padding", 1),
        compression_threshold=DCCP.get("compression_threshold", 8),
        initial_uncompressed_keep_length=DCCP.get("initial_uncompressed_keep_length", 2),
        compress_layer_ids=compress_ids_test
    )

    try:
        env = CompressionEnv(
            compressor_config=test_env_compressor_config,
            llm_model_name_or_path=LCONF["model_name_or_path"],
            tokenizer_name_or_path=LCONF.get("tokenizer_name_or_path", LCONF["model_name_or_path"]),
            llm_device=LCONF["device"],
            training_params=TRAINING_PARAMS,
            llm_config_params=LCONF
        )
    except Exception as e:
        print(f"Error during CompressionEnv test initialization: {e}")
        raise

    num_test_steps_env = 20
    env.reset()
    print(f"\n--- Running {num_test_steps_env} test steps for CompressionEnv ---")
    for step in range(num_test_steps_env):
        print(f"\nTest Step {step + 1}/{num_test_steps_env}")
        result = env.generate_one_step()
        if result is None:
            print("  generate_one_step returned None, breaking test.")
            break
        loss_tensor, done, info = result
        if loss_tensor is not None:
            print(f"  Loss: {loss_tensor.item():.6f}")
            print(f"  Info: {info}")
        else:
            print("  Loss tensor is None.")
        if done:
            print(f"Episode finished at step {step + 1}. Resetting environment.")
            env.reset()
    env.close()
    print("\nCompression Environment test finished.")