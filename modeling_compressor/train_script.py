# train_script.py
from typing import List, Optional, Dict, Any
import torch
import torch.optim as optim
import numpy as np
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 导入配置和自定义模块
from config_rl import TRAINING_PARAMS, LLM_CONFIG, DEFAULT_COMPRESSOR_CONFIG_PARAMS, DATASET_CONFIG, CURRICULUM_LEARNING_CONFIG #
from untis import CompressorConfig #
from compression_env import CompressionEnv
# model.py 和 CCache.py 会被 compression_env.py 和 CompressorConfig 间接使用

def plot_losses(step_losses: List[float], save_path: str = "training_losses.png", window_size: int = 100, stage_num: Optional[int] = None): #
    """绘制训练损失曲线和滑动平均损失曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(step_losses, label="Loss per Step", alpha=0.3)
    
    if len(step_losses) >= window_size:
        moving_avg = np.convolve(np.array(step_losses), np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(step_losses)), moving_avg, 
                 label=f"Moving Average Loss (window {window_size})", linewidth=2, color='red') #
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss (e.g., KL Divergence or MSE)")
    title = "KVCompressor Training Loss Over Steps"
    if stage_num is not None:
        title = f"KVCompressor Training Loss (Stage {stage_num}) Over Steps"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    # print(f"Loss plot saved to {save_path}")

def main_compressor_training_loop():
    print("Starting KVCompressor Training Loop with Dataset Support, Curriculum Learning & TensorBoard...") #
    start_time = datetime.now()
    # 强制使用CPU进行测试以避免内存问题
    # device = torch.device("cpu") # 强制使用CPU进行测试
    device = torch.device(LLM_CONFIG["device"]) #
    print(f"Using device: {device}")

    # --- 1. 初始化 TensorBoard Writer ---
    log_dir_base = TRAINING_PARAMS.get("TENSORBOARD_LOG_DIR", "runs/compressor_curriculum") #
    log_dir = os.path.join(log_dir_base, 'test') #
    writer = SummaryWriter(log_dir=log_dir) #
    model_save_dir = TRAINING_PARAMS.get("MODEL_SAVE_DIR","./ckp")
    output_dir = model_save_dir
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- 2. 课程学习阶段管理 ---
    cl_config = TRAINING_PARAMS.get("curriculum_learning_config", CURRICULUM_LEARNING_CONFIG) #
    if not cl_config.get("enabled", False):
        print("Curriculum learning disabled. Running a single training stage.")
        cl_config["total_stages"] = 1
        cl_config["stage_1"] = {
            "enabled": True,
            "duration_steps": TRAINING_PARAMS["NUM_TRAINING_STEPS"], #
            "dataset_args_override": TRAINING_PARAMS.get("dataset_args", DATASET_CONFIG), #
            "training_params_override": {} 
        }

    total_steps_overall = 0
    all_step_losses_overall = [] 
    prev_model_save_path: Optional[str] = None
    
    # 全局预训练模型加载 (如果配置了，只在所有阶段开始前加载一次)
    # 后续阶段间的模型加载由 prev_model_save_path 控制
    global_pretrained_compressor_path = TRAINING_PARAMS.get("LOAD_PRETRAINED_COMPRESSOR_PATH", None) #
    initial_k_compressor_state_dict = None
    initial_v_compressor_state_dict = None
    initial_optimizer_state_dict = None

    if global_pretrained_compressor_path and os.path.exists(global_pretrained_compressor_path): #
        print(f"Attempting to load globally configured pretrained compressor weights from: {global_pretrained_compressor_path}")
        try:
            checkpoint = torch.load(global_pretrained_compressor_path, map_location=device) #
            if 'k_compressor_state_dict' in checkpoint:
                initial_k_compressor_state_dict = checkpoint['k_compressor_state_dict']
            if 'v_compressor_state_dict' in checkpoint:
                initial_v_compressor_state_dict = checkpoint['v_compressor_state_dict']
            if 'optimizer_state_dict' in checkpoint: # 也预加载优化器状态
                initial_optimizer_state_dict = checkpoint['optimizer_state_dict']
            print("Global pretrained KVCompressor weights (and potentially optimizer state) staged for loading.")
        except Exception as e:
            print(f"Warning: Could not load globally pretrained compressor weights from {global_pretrained_compressor_path}: {e}")
    else:
        print("No global pretrained compressor model specified or found. Each stage will manage its model loading.")


    # --- 外部循环：管理训练阶段 ---
    for stage_num in range(1, cl_config.get("total_stages", 1) + 1):
        stage_key = f"stage_{stage_num}"
        stage_config = cl_config.get(stage_key)

        if not stage_config or not stage_config.get("enabled", False if stage_num > 1 else True): # 默认启用stage_1
            print(f"Skipping curriculum learning stage {stage_num} as it's not defined or not enabled.")
            continue

        stage_duration_steps = stage_config["duration_steps"]
        
        current_stage_training_params = TRAINING_PARAMS.copy()
        current_stage_training_params.update(stage_config.get("training_params_override", {}))
        
        current_stage_dataset_args = TRAINING_PARAMS.get("dataset_args", DATASET_CONFIG).copy() #
        if stage_config.get("dataset_args_override") is not None:
             current_stage_dataset_args.update(stage_config.get("dataset_args_override"))
        
        print(f"\n--- Starting Curriculum Learning Stage {stage_num}/{cl_config['total_stages']} ---")
        print(f"  Duration: {stage_duration_steps} steps")
        print(f"  Effective LR for this stage: {current_stage_training_params['COMPRESSOR_LEARNING_RATE']}")
        print(f"  Dataset args for this stage: name='{current_stage_dataset_args.get('hf_dataset_name', 'N/A')}', "
              f"config='{current_stage_dataset_args.get('hf_dataset_config_name', 'N/A')}', "
              f"split='{current_stage_dataset_args.get('hf_split', 'N/A')}', "
              f"max_samples='{current_stage_dataset_args.get('max_samples_to_load', 'All')}', "
              f"max_tok_len='{current_stage_dataset_args.get('max_tokenized_length', 'Default')}'")

        try:
            from transformers import AutoConfig #
            llm_true_config = AutoConfig.from_pretrained(LLM_CONFIG["model_name_or_path"]) #
            cfg_layer_nums = llm_true_config.num_hidden_layers #
            cfg_kv_heads = getattr(llm_true_config, 'num_key_value_heads', llm_true_config.num_attention_heads) #
            cfg_input_dim = (llm_true_config.hidden_size // llm_true_config.num_attention_heads) * cfg_kv_heads #
        except Exception as e:
            print(f"Warning: Could not auto-config dimensions from LLM ({e}). Using defaults from config.") #
            cfg_layer_nums = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("layer_nums", 2) #
            cfg_kv_heads = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kv_head_dim", 2) #
            cfg_input_dim = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("input_dim", 128) #

        skip_layers_cfg = current_stage_training_params.get("skip_first_n_layers_for_compression", 
                                 DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("skip_first_n_layers_for_compression", 4)) #
        if cfg_layer_nums <= skip_layers_cfg:
            stage_compress_ids = [cfg_layer_nums - 1] if cfg_layer_nums > 0 else []
        else:
            stage_compress_ids = list(range(skip_layers_cfg, cfg_layer_nums))

        compressor_cfg_obj = CompressorConfig(
            input_dim=cfg_input_dim,
            reduction_factor=current_stage_training_params.get("reduction_factor", DEFAULT_COMPRESSOR_CONFIG_PARAMS["reduction_factor"]), #
            output_seq_len=current_stage_training_params.get("output_seq_len", DEFAULT_COMPRESSOR_CONFIG_PARAMS["output_seq_len"]), #
            num_attention_heads=current_stage_training_params.get("num_attention_heads", DEFAULT_COMPRESSOR_CONFIG_PARAMS["num_attention_heads"]), #
            use_mixed_precision=current_stage_training_params.get("use_mixed_precision", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("use_mixed_precision", False)), #
            torch_dtype=current_stage_training_params.get("torch_dtype", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("torch_dtype", "torch.float32")), #
            kv_head_dim=cfg_kv_heads, 
            layer_nums=cfg_layer_nums, 
            kernel_size=current_stage_training_params.get("kernel_size", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kernel_size", 3)), #
            padding=current_stage_training_params.get("padding", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("padding", 1)), #
            compression_threshold=current_stage_training_params.get("compression_threshold", DEFAULT_COMPRESSOR_CONFIG_PARAMS["compression_threshold"]), #
            initial_uncompressed_keep_length=current_stage_training_params.get("initial_uncompressed_keep_length", DEFAULT_COMPRESSOR_CONFIG_PARAMS["initial_uncompressed_keep_length"]), #
            compress_layer_ids=stage_compress_ids
        )

        try:
            environment = CompressionEnv(
                compressor_config=compressor_cfg_obj,
                llm_model_name_or_path=LLM_CONFIG["model_name_or_path"], #
                tokenizer_name_or_path=LLM_CONFIG.get("tokenizer_name_or_path", LLM_CONFIG["model_name_or_path"]), #
                llm_device=LLM_CONFIG["device"], #
                training_params=current_stage_training_params,
                llm_config_params=LLM_CONFIG, #
                current_stage_dataset_args=current_stage_dataset_args
            )
        except Exception as e:
            print(f"FATAL: Error initializing CompressionEnv for stage {stage_num}: {e}")
            import traceback; traceback.print_exc(); writer.close(); return

        params_to_optimize = list(environment.comp_cache.k_compressor.parameters()) + \
                             list(environment.comp_cache.v_compressor.parameters()) #
        
        compressor_optimizer = optim.Adam( #
            filter(lambda p: p.requires_grad, params_to_optimize), 
            lr=current_stage_training_params["COMPRESSOR_LEARNING_RATE"] #
        )

        # 模型和优化器状态加载逻辑
        path_to_load_for_this_stage = None
        if stage_num == 1 and initial_k_compressor_state_dict: # 全局预训练模型用于第一阶段
            path_to_load_for_this_stage = global_pretrained_compressor_path # 仅用于打印信息
            try:
                environment.comp_cache.k_compressor.load_state_dict(initial_k_compressor_state_dict) #
                if initial_v_compressor_state_dict:
                    environment.comp_cache.v_compressor.load_state_dict(initial_v_compressor_state_dict) #
                else: # 如果v和k用同一个模型权重
                    environment.comp_cache.v_compressor.load_state_dict(initial_k_compressor_state_dict)
                print(f"Stage {stage_num}: Loaded K/V compressor weights from global pretrained model: {global_pretrained_compressor_path}")
                if initial_optimizer_state_dict and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_GLOBAL", False): #
                    compressor_optimizer.load_state_dict(initial_optimizer_state_dict)
                    print("Global optimizer state loaded for Stage 1.")
            except Exception as e:
                print(f"Warning: Stage {stage_num} - could not load weights from global pretrained model {global_pretrained_compressor_path}: {e}")
        elif stage_num > 1 and prev_model_save_path and os.path.exists(prev_model_save_path): # 后续阶段加载上一阶段模型
            path_to_load_for_this_stage = prev_model_save_path
            try:
                checkpoint = torch.load(prev_model_save_path, map_location=device) #
                environment.comp_cache.k_compressor.load_state_dict(checkpoint['k_compressor_state_dict']) #
                environment.comp_cache.v_compressor.load_state_dict(checkpoint['v_compressor_state_dict']) #
                print(f"Stage {stage_num}: Loaded K/V compressor weights from previous stage: {prev_model_save_path}")
                if 'optimizer_state_dict' in checkpoint and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_FROM_PREV_STAGE", False): #
                     # 需要在重新创建优化器之前加载，或者加载后更新优化器参数组和lr
                    compressor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded from previous stage.")
            except Exception as e:
                print(f"Warning: Stage {stage_num} - could not load weights from {prev_model_save_path}: {e}")
        else:
            print(f"Stage {stage_num}: Starting with fresh or existing (if not first stage and no prev_model) compressor weights.")


        lr_scheduler = None
        if current_stage_training_params.get("USE_LR_SCHEDULER", False): #
            lr_scheduler = torch.optim.lr_scheduler.StepLR( #
                compressor_optimizer, 
                step_size=current_stage_training_params.get("LR_SCHEDULER_STEP_SIZE", 5000), #
                gamma=current_stage_training_params.get("LR_SCHEDULER_GAMMA", 0.5) #
            )

        environment.reset()
        print(f"  Starting stage {stage_num} training loop for {stage_duration_steps} steps...")
        for step_in_stage in range(1, stage_duration_steps + 1):
            current_global_step = total_steps_overall + step_in_stage
            
            result = environment.generate_one_step() #
            if result is None:
                print(f"Warning: Stage {stage_num}, Step {step_in_stage} (Global {current_global_step}) env.generate_one_step() returned None. Resetting.")
                environment.reset() 
                continue

            loss_tensor, done, info = result #
            current_loss_value = float('nan') #

            if loss_tensor is not None:
                current_loss_value = loss_tensor.item() #
                if loss_tensor.requires_grad: #
                    compressor_optimizer.zero_grad() #
                    loss_tensor.backward() #
                    if current_stage_training_params.get("GRADIENT_CLIP_NORM", 0) > 0: #
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, current_stage_training_params["GRADIENT_CLIP_NORM"]) #
                    compressor_optimizer.step() #
                    if lr_scheduler: #
                        lr_scheduler.step() 
            else:
                print(f"Critical Warning: loss_tensor is None at Stage {stage_num}, Step {step_in_stage}. Skipping update.")
                if done: environment.reset()
                continue
            
            all_step_losses_overall.append(current_loss_value) #

            writer.add_scalar(f'Loss/step_stage_{stage_num}', current_loss_value, current_global_step) #
            writer.add_scalar('Loss/step_overall', current_loss_value, current_global_step)
            if 'compressed_logits_entropy' in info: #
                writer.add_scalar(f'Entropy/compressed_logits_stage_{stage_num}', info['compressed_logits_entropy'], current_global_step) #
            if 'ref_logits_entropy' in info: #
                writer.add_scalar(f'Entropy/reference_logits_stage_{stage_num}', info['ref_logits_entropy'], current_global_step) #
            if info.get('compression_performed_this_step', False): #
                writer.add_scalar(f'Compression/performed_stage_{stage_num}', 1, current_global_step) #
            else:
                writer.add_scalar(f'Compression/performed_stage_{stage_num}', 0, current_global_step) #
            writer.add_scalar('LearningRate/compressor', compressor_optimizer.param_groups[0]['lr'], current_global_step) #
            
            if current_global_step % current_stage_training_params["LOG_FREQ_STEPS"] == 0: #
                 avg_loss_recent_overall = np.mean(all_step_losses_overall[-100:]) if all_step_losses_overall else float('nan') #
                 print(f"S{stage_num} Ep {environment._current_episode_total_tokens_generated // current_stage_training_params.get('MAX_TOKENS_PER_EPISODE',1)+1 } St {step_in_stage}/{stage_duration_steps} (Glbl {current_global_step}): "
                       f"Loss: {current_loss_value:.5f}, AvgLoss(100): {avg_loss_recent_overall:.5f}, LR: {compressor_optimizer.param_groups[0]['lr']:.1e}, " #
                       f"TokMain: {info.get('predicted_token_main', 'N/A')}") #

            if done: #
                # print(f"  Done flag received at Stage {stage_num}, Step {step_in_stage}. Resetting environment.")
                environment.reset() 
            
            if current_global_step % current_stage_training_params["COMPRESSOR_MODEL_SAVE_FREQ_STEPS"] == 0: #
                periodic_model_save_path = os.path.join(model_save_dir, f"kv_compressor_stage{stage_num}_glbStep{current_global_step}.pth") #
                torch.save({ #
                    'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), #
                    'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), #
                    'optimizer_state_dict': compressor_optimizer.state_dict(), 'step': current_global_step, #
                    'stage_num': stage_num, 'loss': current_loss_value #
                }, periodic_model_save_path)
                print(f"KVCompressor model periodically saved to {periodic_model_save_path}") #
                if len(all_step_losses_overall) > 10: #
                    plot_losses(all_step_losses_overall, save_path=os.path.join(output_dir, f"losses_glbStep{current_global_step}.png"), stage_num=stage_num) #
        
        total_steps_overall += stage_duration_steps
        stage_end_model_save_path = os.path.join(model_save_dir, f"kv_compressor_stage{stage_num}_final_step{total_steps_overall}.pth") #
        torch.save({ #
            'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), #
            'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), #
            'optimizer_state_dict': compressor_optimizer.state_dict(), #
            'step': total_steps_overall, 
            'stage_num': stage_num, 
            'final_stage_loss_avg': np.mean(all_step_losses_overall[-stage_duration_steps:]) if all_step_losses_overall and stage_duration_steps > 0 else float('nan') #
        }, stage_end_model_save_path)
        prev_model_save_path = stage_end_model_save_path 
        print(f"KVCompressor model for end of stage {stage_num} saved to {prev_model_save_path}")
        print(f"--- Stage {stage_num} Finished ({stage_duration_steps} steps completed) ---")

    end_time = datetime.now() #
    print(f"\nKVCompressor training finished in {end_time - start_time}.") #
    final_model_path = os.path.join(model_save_dir, "kv_compressor_final_overall.pth") #
    torch.save({ #
        'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), #
        'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), #
        'optimizer_state_dict': compressor_optimizer.state_dict(), #
        'step': total_steps_overall, #
        'final_avg_loss': np.mean(all_step_losses_overall[-1000:]) if all_step_losses_overall else float('nan') #
    }, final_model_path)
    print(f"Final KVCompressor model saved to {final_model_path}") #
    if all_step_losses_overall: plot_losses(all_step_losses_overall, save_path=os.path.join(output_dir, "losses_final_overall.png"), stage_num="Overall") #
    
    writer.close() #
    environment.close() #
    print("Environment closed.") #


if __name__ == "__main__":
    seed = TRAINING_PARAMS.get("RANDOM_SEED", 42) #
    if seed is not None:
        torch.manual_seed(seed) #
        np.random.seed(seed) #
        random.seed(seed) #
        print(f"Using random seed: {seed}") #
    try:
        main_compressor_training_loop() #
    except Exception as e:
        print(f"An error occurred during the KVCompressor training loop: {e}") #
        import traceback; traceback.print_exc() #