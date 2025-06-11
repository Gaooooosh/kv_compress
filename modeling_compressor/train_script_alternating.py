# train_script_alternating_strict_grad.py
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
from untils import CompressorConfig #
from compression_env import CompressionEnv #

def plot_losses(step_losses: List[float], save_path: str = "training_losses.png", window_size: int = 100, stage_num: Optional[int] = None, title_suffix: str = ""): #
    """绘制训练损失曲线和滑动平均损失曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(step_losses, label="Loss per Step", alpha=0.3)
    
    if len(step_losses) >= window_size:
        loss_array = np.array(step_losses)
        if loss_array.ndim == 0: 
            loss_array = np.array([loss_array.item()]) if isinstance(loss_array, torch.Tensor) else np.array([loss_array])
        if len(loss_array) >= window_size:
            moving_avg = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size-1, len(loss_array)), moving_avg, 
                     label=f"Moving Average Loss (window {window_size})", linewidth=2, color='red')
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss (e.g., KL Divergence or MSE)")
    title = f"KVCompressor Training Loss{title_suffix}"
    if stage_num is not None:
        title = f"KVCompressor Training Loss (Stage {stage_num}{title_suffix}) Over Steps"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def set_requires_grad(module: torch.nn.Module, requires_grad: bool): #
    """辅助函数，用于设置模块所有参数的requires_grad属性"""
    for param in module.parameters():
        param.requires_grad = requires_grad

def main_compressor_training_loop():
    print("Starting KVCompressor Training Loop (Strict Alternating K/V, Dataset Support, Curriculum Learning & TensorBoard)...")
    start_time = datetime.now()
    device = torch.device(LLM_CONFIG["device"])
    print(f"Using device: {device}")

    # --- 1. 初始化 TensorBoard Writer ---
    log_dir_base = TRAINING_PARAMS.get("TENSORBOARD_LOG_DIR", "runs/compressor_alternating_strict")
    log_dir = os.path.join(log_dir_base, "test")
    writer = SummaryWriter(log_dir=log_dir)
    model_save_dir = TRAINING_PARAMS.get("MODEL_SAVE_DIR","./ckp")
    output_dir = model_save_dir
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- 2. 课程学习阶段管理 ---
    cl_config = TRAINING_PARAMS.get("curriculum_learning_config", CURRICULUM_LEARNING_CONFIG)
    if not cl_config.get("enabled", False):
        print("Curriculum learning disabled. Running a single training stage.")
        cl_config["total_stages"] = 1
        cl_config["stage_1"] = {
            "enabled": True,
            "duration_steps": TRAINING_PARAMS["NUM_TRAINING_STEPS"],
            "dataset_args_override": TRAINING_PARAMS.get("dataset_args", DATASET_CONFIG),
            "training_params_override": {}
        }

    total_steps_overall = 0
    all_step_losses_overall = []
    prev_model_save_path: Optional[str] = None
    global_pretrained_compressor_path = TRAINING_PARAMS.get("LOAD_PRETRAINED_COMPRESSOR_PATH", None)
    initial_k_compressor_state_dict = None
    initial_v_compressor_state_dict = None
    initial_k_optimizer_state_dict = None 
    initial_v_optimizer_state_dict = None

    if global_pretrained_compressor_path and os.path.exists(global_pretrained_compressor_path):
        print(f"Attempting to load globally configured pretrained compressor weights from: {global_pretrained_compressor_path}")
        try:
            checkpoint = torch.load(global_pretrained_compressor_path, map_location=device)
            if 'k_compressor_state_dict' in checkpoint:
                initial_k_compressor_state_dict = checkpoint['k_compressor_state_dict']
            if 'v_compressor_state_dict' in checkpoint:
                initial_v_compressor_state_dict = checkpoint['v_compressor_state_dict']
            if 'k_optimizer_state_dict' in checkpoint and TRAINING_PARAMS.get("LOAD_OPTIMIZER_STATE_GLOBAL", False) :
                 initial_k_optimizer_state_dict = checkpoint['k_optimizer_state_dict']
            if 'v_optimizer_state_dict' in checkpoint and TRAINING_PARAMS.get("LOAD_OPTIMIZER_STATE_GLOBAL", False) :
                 initial_v_optimizer_state_dict = checkpoint['v_optimizer_state_dict']
            print("Global pretrained K/V Compressor weights (and potentially optimizer states) staged for loading.")
        except Exception as e:
            print(f"Warning: Could not load globally pretrained compressor weights from {global_pretrained_compressor_path}: {e}")
    else:
        print("No global pretrained compressor model specified or found.")

    k_optimizer: Optional[optim.Adam] = None # 在外部定义，以便在阶段间可能保持（如果适用）
    v_optimizer: Optional[optim.Adam] = None
    k_params: List[torch.nn.Parameter] = []
    v_params: List[torch.nn.Parameter] = []


    for stage_num in range(1, cl_config.get("total_stages", 1) + 1):
        stage_key = f"stage_{stage_num}"
        stage_config = cl_config.get(stage_key)
        if not stage_config or not stage_config.get("enabled", False if stage_num > 1 else True):
            print(f"Skipping curriculum learning stage {stage_num} as it's not defined or not enabled.")
            continue

        stage_duration_steps = stage_config["duration_steps"]
        current_stage_training_params = TRAINING_PARAMS.copy()
        current_stage_training_params.update(stage_config.get("training_params_override", {}))
        current_stage_dataset_args = TRAINING_PARAMS.get("dataset_args", DATASET_CONFIG).copy()
        if stage_config.get("dataset_args_override") is not None:
             current_stage_dataset_args.update(stage_config.get("dataset_args_override"))
        
        print(f"\n--- Starting Curriculum Learning Stage {stage_num}/{cl_config['total_stages']} (Strict Alternating K/V) ---")
        print(f"  Duration: {stage_duration_steps} steps")
        print(f"  Effective LR for K: {current_stage_training_params.get('K_COMPRESSOR_LEARNING_RATE', current_stage_training_params['COMPRESSOR_LEARNING_RATE'])}")
        print(f"  Effective LR for V: {current_stage_training_params.get('V_COMPRESSOR_LEARNING_RATE', current_stage_training_params['COMPRESSOR_LEARNING_RATE'])}")
        # ... (打印数据集信息)

        try:
            from transformers import AutoConfig
            llm_true_config = AutoConfig.from_pretrained(LLM_CONFIG["model_name_or_path"])
            cfg_layer_nums = llm_true_config.num_hidden_layers
            cfg_kv_heads = getattr(llm_true_config, 'num_key_value_heads', llm_true_config.num_attention_heads)
            cfg_input_dim = (llm_true_config.hidden_size // llm_true_config.num_attention_heads) * cfg_kv_heads
        except Exception as e:
            cfg_layer_nums = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("layer_nums", 2)
            cfg_kv_heads = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kv_head_dim", 2)
            cfg_input_dim = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("input_dim", 128)

        skip_layers_cfg = current_stage_training_params.get("skip_first_n_layers_for_compression", 
                                 DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("skip_first_n_layers_for_compression", 4))
        if cfg_layer_nums <= skip_layers_cfg:
            stage_compress_ids = [cfg_layer_nums - 1] if cfg_layer_nums > 0 else []
        else:
            stage_compress_ids = list(range(skip_layers_cfg, cfg_layer_nums))
        
        compressor_cfg_obj = CompressorConfig(
            input_dim=cfg_input_dim,
            reduction_factor=current_stage_training_params.get("reduction_factor", DEFAULT_COMPRESSOR_CONFIG_PARAMS["reduction_factor"]),
            output_seq_len=current_stage_training_params.get("output_seq_len", DEFAULT_COMPRESSOR_CONFIG_PARAMS["output_seq_len"]),
            num_attention_heads=current_stage_training_params.get("num_attention_heads", DEFAULT_COMPRESSOR_CONFIG_PARAMS["num_attention_heads"]),
            use_mixed_precision=current_stage_training_params.get("use_mixed_precision", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("use_mixed_precision", False)),
            torch_dtype=current_stage_training_params.get("torch_dtype", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("torch_dtype", "torch.float32")),
            kv_head_dim=cfg_kv_heads, 
            layer_nums=cfg_layer_nums, 
            kernel_size=current_stage_training_params.get("kernel_size", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kernel_size", 3)),
            padding=current_stage_training_params.get("padding", DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("padding", 1)),
            compression_threshold=current_stage_training_params.get("compression_threshold", DEFAULT_COMPRESSOR_CONFIG_PARAMS["compression_threshold"]),
            initial_uncompressed_keep_length=current_stage_training_params.get("initial_uncompressed_keep_length", DEFAULT_COMPRESSOR_CONFIG_PARAMS["initial_uncompressed_keep_length"]),
            compress_layer_ids=stage_compress_ids
        )

        try:
            environment = CompressionEnv(
                compressor_config=compressor_cfg_obj,
                llm_model_name_or_path=LLM_CONFIG["model_name_or_path"],
                tokenizer_name_or_path=LLM_CONFIG.get("tokenizer_name_or_path", LLM_CONFIG["model_name_or_path"]),
                llm_device=LLM_CONFIG["device"],
                training_params=current_stage_training_params,
                llm_config_params=LLM_CONFIG,
                current_stage_dataset_args=current_stage_dataset_args
            )
        except Exception as e:
            print(f"FATAL: Error initializing CompressionEnv for stage {stage_num}: {e}")
            import traceback; traceback.print_exc(); writer.close(); return

        # 初始化或更新优化器
        k_params = list(filter(lambda p: p.requires_grad, environment.comp_cache.k_compressor.parameters()))
        v_params = list(filter(lambda p: p.requires_grad, environment.comp_cache.v_compressor.parameters()))
        if not k_params or not v_params:
            print("FATAL: K or V compressor has no trainable parameters after env init for this stage."); writer.close(); return

        k_lr = current_stage_training_params.get("K_COMPRESSOR_LEARNING_RATE", current_stage_training_params["COMPRESSOR_LEARNING_RATE"])
        v_lr = current_stage_training_params.get("V_COMPRESSOR_LEARNING_RATE", current_stage_training_params["COMPRESSOR_LEARNING_RATE"])

        if k_optimizer is None or stage_num > 1: # 首次或新阶段，重新创建
            k_optimizer = optim.Adam(k_params, lr=k_lr)
            print(f"K_Optimizer (re)initialized for Stage {stage_num} with LR: {k_lr}")
        else: # 更新现有优化器的学习率
            for g in k_optimizer.param_groups: g['lr'] = k_lr
            print(f"K_Optimizer LR updated for Stage {stage_num} to: {k_lr}")
        
        if v_optimizer is None or stage_num > 1:
            v_optimizer = optim.Adam(v_params, lr=v_lr)
            print(f"V_Optimizer (re)initialized for Stage {stage_num} with LR: {v_lr}")
        else:
            for g in v_optimizer.param_groups: g['lr'] = v_lr
            print(f"V_Optimizer LR updated for Stage {stage_num} to: {v_lr}")
        
        # 模型和优化器状态加载
        if stage_num == 1:
            if initial_k_compressor_state_dict:
                try: environment.comp_cache.k_compressor.load_state_dict(initial_k_compressor_state_dict)
                except Exception as e: print(f"Warning: Stage {stage_num} - could not load K-weights from global: {e}")
                else: print(f"Stage {stage_num}: Loaded K-compressor weights from global: {global_pretrained_compressor_path}")
            if initial_v_compressor_state_dict:
                try: environment.comp_cache.v_compressor.load_state_dict(initial_v_compressor_state_dict)
                except Exception as e: print(f"Warning: Stage {stage_num} - could not load V-weights from global: {e}")
                else: print(f"Stage {stage_num}: Loaded V-compressor weights from global: {global_pretrained_compressor_path}")
            if initial_k_optimizer_state_dict and k_optimizer and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_GLOBAL", False):
                try: k_optimizer.load_state_dict(initial_k_optimizer_state_dict)
                except Exception as e: print(f"Warning: Stage {stage_num} - could not load K-optim state: {e}")
                else: print("Global K-optimizer state loaded.")
            if initial_v_optimizer_state_dict and v_optimizer and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_GLOBAL", False):
                try: v_optimizer.load_state_dict(initial_v_optimizer_state_dict)
                except Exception as e: print(f"Warning: Stage {stage_num} - could not load V-optim state: {e}")
                else: print("Global V-optimizer state loaded.")
        elif prev_model_save_path and os.path.exists(prev_model_save_path):
            try:
                checkpoint = torch.load(prev_model_save_path, map_location=device)
                environment.comp_cache.k_compressor.load_state_dict(checkpoint['k_compressor_state_dict'])
                environment.comp_cache.v_compressor.load_state_dict(checkpoint['v_compressor_state_dict'])
                print(f"Stage {stage_num}: Loaded K/V compressor weights from previous stage: {prev_model_save_path}")
                if 'k_optimizer_state_dict' in checkpoint and k_optimizer and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_FROM_PREV_STAGE", True): # Default true
                    k_optimizer.load_state_dict(checkpoint['k_optimizer_state_dict'])
                    for g in k_optimizer.param_groups: g['lr'] = k_lr # Ensure LR is current stage's
                if 'v_optimizer_state_dict' in checkpoint and v_optimizer and current_stage_training_params.get("LOAD_OPTIMIZER_STATE_FROM_PREV_STAGE", True): # Default true
                    v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
                    for g in v_optimizer.param_groups: g['lr'] = v_lr # Ensure LR is current stage's
                if ('k_optimizer_state_dict' in checkpoint or 'v_optimizer_state_dict' in checkpoint) and \
                   current_stage_training_params.get("LOAD_OPTIMIZER_STATE_FROM_PREV_STAGE", True):
                    print("Optimizer states loaded from previous stage and LRs reset.")
            except Exception as e:
                print(f"Warning: Stage {stage_num} - could not load K/V or optimizer from {prev_model_save_path}: {e}")
        else:
             print(f"Stage {stage_num}: Starting with fresh or existing compressor weights (no previous stage model found).")

        lr_scheduler_k, lr_scheduler_v = None, None
        if current_stage_training_params.get("USE_LR_SCHEDULER", False):
            scheduler_step_size = int(current_stage_training_params.get("LR_SCHEDULER_STEP_SIZE", 5000))
            scheduler_gamma = current_stage_training_params.get("LR_SCHEDULER_GAMMA", 0.5)
            if scheduler_step_size > 0:
                if k_optimizer: lr_scheduler_k = torch.optim.lr_scheduler.StepLR(k_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
                if v_optimizer: lr_scheduler_v = torch.optim.lr_scheduler.StepLR(v_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        environment.reset()
        print(f"  Starting stage {stage_num} training loop for {stage_duration_steps} steps...")
        
        for step_in_stage in range(1, stage_duration_steps + 1):
            current_global_step = total_steps_overall + step_in_stage
            
            # 1. 确保在计算loss前，两个压缩器的参数都允许计算梯度，以便loss_tensor连接到两者
            set_requires_grad(environment.comp_cache.k_compressor, True)
            set_requires_grad(environment.comp_cache.v_compressor, True)

            result = environment.generate_one_step()
            if result is None:
                print(f"Warning: Stage {stage_num}, Step {step_in_stage} (Global {current_global_step}) env.generate_one_step() returned None. Resetting.")
                environment.reset() 
                continue

            loss_tensor, done, info = result
            current_loss_value = float('nan')

            if loss_tensor is not None:
                current_loss_value = loss_tensor.item() 
                if loss_tensor.requires_grad:
                    
                    train_k_this_step = False
                    train_v_this_step = False
                    mode = current_stage_training_params.get("ALTERNATING_TRAINING_MODE", "block")
                    block_size = current_stage_training_params.get("ALTERNATING_BLOCK_SIZE", 1) 

                    if mode == "step":
                        if current_global_step % 2 == 0: train_k_this_step = True
                        else: train_v_this_step = True
                    elif mode == "block":
                        if (current_global_step // block_size) % 2 == 0: train_k_this_step = True
                        else: train_v_this_step = True
                    
                    # --- 严格交替训练的优化步骤 ---
                    if train_k_this_step:
                        set_requires_grad(environment.comp_cache.k_compressor, True)
                        set_requires_grad(environment.comp_cache.v_compressor, False)
                        k_optimizer.zero_grad()
                        loss_tensor.backward()
                        k_conv1_weight = environment.comp_cache.k_compressor.conv1.weight
                        # print(f"  K_Compressor conv1.weight: requires_grad={k_conv1_weight.requires_grad}, grad_fn={k_conv1_weight.grad_fn}, is_leaf={k_conv1_weight.is_leaf}")

                    elif train_v_this_step:
                        set_requires_grad(environment.comp_cache.k_compressor, False)
                        set_requires_grad(environment.comp_cache.v_compressor, True)
                        v_optimizer.zero_grad()
                        loss_tensor.backward()
                        v_conv1_weight = environment.comp_cache.v_compressor.conv1.weight

                    print_grad_this_step = (current_global_step % current_stage_training_params.get("LOG_FREQ_STEPS", 100) == 0) # 例如每LOG_FREQ_STEPS打印一次梯度
                    if print_grad_this_step:
                        print(f"    DEBUG GRADS for Step {current_global_step} (Training {'K' if train_k_this_step else 'V'}):")
                        compressor_to_check = environment.comp_cache.k_compressor if train_k_this_step else environment.comp_cache.v_compressor
                        for name, param in compressor_to_check.named_parameters():
                            if param.grad is not None:
                                grad_abs_mean = param.grad.abs().mean().item()
                                grad_abs_max = param.grad.abs().max().item()
                                print(f"      Param: {name:<30} | Grad Mean Abs: {grad_abs_mean:.2e} | Grad Max Abs: {grad_abs_max:.2e}")
                                if grad_abs_mean == 0.0 and grad_abs_max == 0.0:
                                    print(f"      WARNING: Grad for {name} is all zeros.")
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    print(f"      WARNING: Grad for {name} contains NaN or Inf!")
                                break
                            else:
                                print(f"      Param: {name:<30} | Grad: None (Likely frozen or not in graph)")
            
            elif loss_tensor is not None: 
                 current_loss_value = loss_tensor.item()

            all_step_losses_overall.append(current_loss_value)
            writer.add_scalar(f'Loss/step_stage_{stage_num}', current_loss_value, current_global_step)
            writer.add_scalar('Loss/step_overall', current_loss_value, current_global_step)
            if k_optimizer: writer.add_scalar('LearningRate/k_compressor', k_optimizer.param_groups[0]['lr'], current_global_step) #
            if v_optimizer: writer.add_scalar('LearningRate/v_compressor', v_optimizer.param_groups[0]['lr'], current_global_step) #


            if current_global_step % current_stage_training_params["LOG_FREQ_STEPS"] == 0:
                 avg_loss_recent_overall = np.mean(all_step_losses_overall[-100:]) if all_step_losses_overall else float('nan')
                 training_whom = ""
                 if train_k_this_step: training_whom = "K"
                 elif train_v_this_step: training_whom = "V" # 使用 elif 确保互斥
                 else: training_whom = "None"


                 k_lr_str = f"{k_optimizer.param_groups[0]['lr']:.1e}" if k_optimizer else "N/A"
                 v_lr_str = f"{v_optimizer.param_groups[0]['lr']:.1e}" if v_optimizer else "N/A"

                 print(f"S{stage_num} St {step_in_stage}/{stage_duration_steps} (Glbl {current_global_step}): "
                       f"Loss: {current_loss_value:.5f}, AvgLoss(100): {avg_loss_recent_overall:.5f}, "
                       f"Training: {training_whom}, "
                       f"K_LR: {k_lr_str}, "
                       f"V_LR: {v_lr_str}")
            if done: environment.reset()
            
            # ... (模型保存逻辑与您版本一致，但要保存k和v的优化器状态) ...
            if current_global_step % current_stage_training_params["COMPRESSOR_MODEL_SAVE_FREQ_STEPS"] == 0:
                periodic_model_save_path = os.path.join(model_save_dir, f"kv_compressor_alt_stage{stage_num}_glbStep{current_global_step}.pth")
                save_dict = {
                    'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), 
                    'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), 
                    'step': current_global_step, 'stage_num': stage_num, 'loss': current_loss_value 
                }
                if k_optimizer: save_dict['k_optimizer_state_dict'] = k_optimizer.state_dict()
                if v_optimizer: save_dict['v_optimizer_state_dict'] = v_optimizer.state_dict()
                torch.save(save_dict, periodic_model_save_path)
                print(f"Alternating KVCompressor model periodically saved to {periodic_model_save_path}")
                if len(all_step_losses_overall) > 10:
                    plot_losses(all_step_losses_overall, save_path=os.path.join(output_dir, f"losses_alt_glbStep{current_global_step}.png"), stage_num=stage_num, title_suffix=" (Strict Alternating)")

        total_steps_overall += stage_duration_steps
        stage_end_model_save_path = os.path.join(model_save_dir, f"kv_compressor_alt_stage{stage_num}_final_step{total_steps_overall}.pth")
        final_save_dict = {
            'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), 
            'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), 
            'step': total_steps_overall, 'stage_num': stage_num, 
            'final_stage_loss_avg': np.mean(all_step_losses_overall[-stage_duration_steps:]) if all_step_losses_overall and stage_duration_steps > 0 else float('nan')
        }
        if k_optimizer: final_save_dict['k_optimizer_state_dict'] = k_optimizer.state_dict()
        if v_optimizer: final_save_dict['v_optimizer_state_dict'] = v_optimizer.state_dict()
        torch.save(final_save_dict, stage_end_model_save_path)
        prev_model_save_path = stage_end_model_save_path
        print(f"Alternating KVCompressor model for end of stage {stage_num} saved to {prev_model_save_path}")
        print(f"--- Stage {stage_num} Finished ({stage_duration_steps} steps completed) ---")

    end_time = datetime.now()
    print(f"\nKVCompressor training (Strict Alternating K/V) finished in {end_time - start_time}.")
    final_model_path_overall = os.path.join(model_save_dir, "kv_compressor_alt_final_overall.pth")
    final_overall_save_dict = {
        'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(), 
        'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(), 
        'step': total_steps_overall, 
        'final_avg_loss': np.mean(all_step_losses_overall[-1000:]) if all_step_losses_overall else float('nan')
    }
    if k_optimizer: final_overall_save_dict['k_optimizer_state_dict'] = k_optimizer.state_dict()
    if v_optimizer: final_overall_save_dict['v_optimizer_state_dict'] = v_optimizer.state_dict()
    torch.save(final_overall_save_dict, final_model_path_overall)
    print(f"Final Strict Alternating KVCompressor model saved to {final_model_path_overall}")
    if all_step_losses_overall: 
        plot_losses(all_step_losses_overall, 
                    save_path=os.path.join(output_dir, "losses_alt_final_overall.png"), 
                    stage_num="Overall", title_suffix=" (Strict Alternating)")
    
    writer.close()
    environment.close()
    print("Environment closed.")


if __name__ == "__main__":
    # ... (设置随机种子) ...
    main_compressor_training_loop()