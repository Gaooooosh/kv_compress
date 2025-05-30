# train_script.py
from typing import List
import torch
import torch.optim as optim
import numpy as np
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# 导入配置和自定义模块
from config_rl import DATASET_CONFIG, TRAINING_PARAMS, LLM_CONFIG, DEFAULT_COMPRESSOR_CONFIG_PARAMS
from untis import CompressorConfig
from compression_env import CompressionEnv # 使用我们新设计的环境

def plot_losses(step_losses: List[float], save_path: str = "training_losses.png", window_size: int = 100):
    """绘制训练损失曲线和滑动平均损失曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(step_losses, label="Loss per Step", alpha=0.3)
    
    if len(step_losses) >= window_size:
        moving_avg = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(step_losses)), moving_avg, 
                 label=f"Moving Average Loss (window {window_size})", linewidth=2, color='red')
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss (e.g., KL Divergence or MSE)")
    plt.title("KVCompressor Training Loss Over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    # print(f"Loss plot saved to {save_path}") # Less verbose during training

def main_compressor_training_loop():
    print("Starting KVCompressor Training Loop with Dataset Support & TensorBoard...")
    start_time = datetime.now()
    device = torch.device(LLM_CONFIG["device"])
    print(f"Using device: {device}")

    current_training_params = TRAINING_PARAMS.copy()
    if "dataset_args" not in current_training_params and 'DATASET_CONFIG' in globals():
        current_training_params["dataset_args"] = DATASET_CONFIG # DATASET_CONFIG from config_rl.py
    elif "dataset_args" not in current_training_params:
        current_training_params["dataset_args"] = {}
        print("Warning: No dataset_args found. Env might use default prompt.")

    # --- 2. 初始化 TensorBoard Writer ---
    log_dir_base = current_training_params.get("TENSORBOARD_LOG_DIR", "runs/compressor_experiment")
    current_time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(log_dir_base, current_time_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    # ------------------------------------

    try:
        from transformers import AutoConfig
        llm_true_config = AutoConfig.from_pretrained(LLM_CONFIG["model_name_or_path"])
        cfg_layer_nums = llm_true_config.num_hidden_layers
        cfg_kv_heads = getattr(llm_true_config, 'num_key_value_heads', llm_true_config.num_attention_heads)
        cfg_input_dim = (llm_true_config.hidden_size // llm_true_config.num_attention_heads) * cfg_kv_heads
    except Exception as e:
        print(f"Warning: Could not auto-config dimensions from LLM ({e}). Using defaults from config.")
        cfg_layer_nums = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("layer_nums", 2)
        cfg_kv_heads = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kv_head_dim", 2)
        cfg_input_dim = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("input_dim", 128)

    skip_layers_train = DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("skip_first_n_layers_for_compression", 4)
    if cfg_layer_nums <= skip_layers_train:
        train_compress_ids = [cfg_layer_nums - 1] if cfg_layer_nums > 0 else []
    else:
        train_compress_ids = list(range(skip_layers_train, cfg_layer_nums))
    # print(f"Training: Total LLM Layers={cfg_layer_nums}, Layers to Compress IDs={train_compress_ids}")

    compressor_cfg_obj = CompressorConfig(
        input_dim=cfg_input_dim,
        reduction_factor=DEFAULT_COMPRESSOR_CONFIG_PARAMS["reduction_factor"],
        output_seq_len=DEFAULT_COMPRESSOR_CONFIG_PARAMS["output_seq_len"],
        num_attention_heads=DEFAULT_COMPRESSOR_CONFIG_PARAMS["num_attention_heads"],
        use_mixed_precision=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("use_mixed_precision", False),
        torch_dtype=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("torch_dtype", "torch.float32"),
        kv_head_dim=cfg_kv_heads, 
        layer_nums=cfg_layer_nums, 
        kernel_size=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("kernel_size", 3),
        padding=DEFAULT_COMPRESSOR_CONFIG_PARAMS.get("padding", 1),
        compression_threshold=DEFAULT_COMPRESSOR_CONFIG_PARAMS["compression_threshold"],
        initial_uncompressed_keep_length=DEFAULT_COMPRESSOR_CONFIG_PARAMS["initial_uncompressed_keep_length"],
        compress_layer_ids=train_compress_ids 
    )
    # print("CompressorConfig for training created.")

    try:
        environment = CompressionEnv(
            compressor_config=compressor_cfg_obj,
            llm_model_name_or_path=LLM_CONFIG["model_name_or_path"],
            tokenizer_name_or_path=LLM_CONFIG.get("tokenizer_name_or_path", LLM_CONFIG["model_name_or_path"]),
            llm_device=LLM_CONFIG["device"],
            training_params=current_training_params,
            llm_config_params=LLM_CONFIG
        )
    except Exception as e:
        print(f"FATAL: Error initializing CompressionEnv: {e}")
        import traceback; traceback.print_exc(); writer.close(); return
    # print("CompressionEnv initialized.")

    params_to_optimize = list(environment.comp_cache.k_compressor.parameters()) + \
                         list(environment.comp_cache.v_compressor.parameters())
    num_trainable_params = sum(p.numel() for p in params_to_optimize if p.requires_grad)
    if num_trainable_params == 0:
        print("FATAL: No trainable parameters found in KVCompressors."); writer.close(); return
    # print(f"KVCompressor has {num_trainable_params} trainable parameters.")
    
    compressor_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params_to_optimize), 
        lr=current_training_params["COMPRESSOR_LEARNING_RATE"]
    )
    # print(f"KVCompressor optimizer initialized with LR: {current_training_params['COMPRESSOR_LEARNING_RATE']}.")
    
    # (可选) 学习率调度器
    lr_scheduler = None
    if current_training_params.get("USE_LR_SCHEDULER", False): # 在config中添加这个参数
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            compressor_optimizer, 
            step_size=current_training_params.get("LR_SCHEDULER_STEP_SIZE", 5000), 
            gamma=current_training_params.get("LR_SCHEDULER_GAMMA", 0.5)
        )
        print(f"Using LR Scheduler: StepLR with step_size={current_training_params.get('LR_SCHEDULER_STEP_SIZE', 5000)}, gamma={current_training_params.get('LR_SCHEDULER_GAMMA', 0.5)}")


    load_compressor_path = current_training_params.get("LOAD_PRETRAINED_COMPRESSOR_PATH", None)
    if load_compressor_path and os.path.exists(load_compressor_path):
        print(f"Loading pretrained KVCompressor weights from: {load_compressor_path}")
        try:
            checkpoint = torch.load(load_compressor_path, map_location=device)
            if 'k_compressor_state_dict' in checkpoint:
                environment.comp_cache.k_compressor.load_state_dict(checkpoint['k_compressor_state_dict'])
            if 'v_compressor_state_dict' in checkpoint:
                environment.comp_cache.v_compressor.load_state_dict(checkpoint['v_compressor_state_dict'])
            print("Pretrained KVCompressor weights loaded.")
        except Exception as e:
            print(f"Warning: Could not load pretrained compressor weights from {load_compressor_path}: {e}")


    all_step_losses = []
    output_dir = current_training_params.get("OUTPUT_DIR", "compressor_training_output_dataset")
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join(output_dir, "saved_compressor_models")
    os.makedirs(model_save_dir, exist_ok=True)

    environment.reset() 
    print(f"\nStarting KVCompressor training for {current_training_params['NUM_TRAINING_STEPS']} steps...")
    
    for step in range(1, current_training_params["NUM_TRAINING_STEPS"] + 1):
        result = environment.generate_one_step()
        if result is None:
            print(f"Warning: Step {step} environment.generate_one_step() returned None. Attempting reset and continue.")
            environment.reset() 
            continue

        loss_tensor, done, info = result

        current_loss_value = float('nan') # 默认值
        if loss_tensor is not None:
            current_loss_value = loss_tensor.item()
            if loss_tensor.requires_grad:
                compressor_optimizer.zero_grad()
                loss_tensor.backward()
                if current_training_params.get("GRADIENT_CLIP_NORM", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, current_training_params["GRADIENT_CLIP_NORM"])
                compressor_optimizer.step()
                
                # 如果使用了学习率调度器，在此更新
                if lr_scheduler:
                    lr_scheduler.step()
            # else:
            #     print(f"Step {step}: Loss tensor does not require grad.")
        else:
            print(f"Critical Warning: loss_tensor is None at step {step}. Skipping update.")
            if done: environment.reset()
            continue
            
        all_step_losses.append(current_loss_value)

        # --- 3. 记录到 TensorBoard ---
        writer.add_scalar('Loss/step', current_loss_value, step)
        if 'compressed_logits_entropy' in info:
            writer.add_scalar('Entropy/compressed_logits', info['compressed_logits_entropy'], step)
        if 'ref_logits_entropy' in info:
            writer.add_scalar('Entropy/reference_logits', info['ref_logits_entropy'], step)
        if info.get('compression_performed_this_step', False): # 只在压缩发生时记录为1，否则为0
            writer.add_scalar('Compression/performed', 1, step)
        else:
            writer.add_scalar('Compression/performed', 0, step)
        
        # 记录学习率 (如果使用了调度器)
        writer.add_scalar('LearningRate/compressor', compressor_optimizer.param_groups[0]['lr'], step)
        # -----------------------------
            
        if step % current_training_params["LOG_FREQ_STEPS"] == 0:
            avg_loss_recent = np.mean(all_step_losses[-100:]) if all_step_losses else float('nan')
            print(f"Step {step}/{current_training_params['NUM_TRAINING_STEPS']}: "
                  f"Loss: {current_loss_value:.6f}, Avg Loss (last 100): {avg_loss_recent:.6f}, "
                  f"LR: {compressor_optimizer.param_groups[0]['lr']:.2e}, " # 打印当前学习率
                  f"PredTokenMain: {info.get('predicted_token_main', 'N/A')}")

        if done:
            environment.reset() 
        
        if step % current_training_params["COMPRESSOR_MODEL_SAVE_FREQ_STEPS"] == 0:
            model_save_path = os.path.join(model_save_dir, f"kv_compressor_step_{step}.pth")
            torch.save({
                'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(),
                'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(),
                'optimizer_state_dict': compressor_optimizer.state_dict(), 'step': step,
                'loss': current_loss_value
            }, model_save_path)
            print(f"KVCompressor model saved to {model_save_path}")
            if len(all_step_losses) > 10:
                plot_losses(all_step_losses, save_path=os.path.join(output_dir, f"losses_step_{step}.png"))

    end_time = datetime.now()
    print(f"\nKVCompressor training finished in {end_time - start_time}.")
    final_model_path = os.path.join(model_save_dir, "kv_compressor_final.pth")
    # ... (保存最终模型逻辑保持不变) ...
    torch.save({
        'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(),
        'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(),
        'optimizer_state_dict': compressor_optimizer.state_dict(), 'step': current_training_params['NUM_TRAINING_STEPS'],
        'final_avg_loss': np.mean(all_step_losses[-1000:]) if all_step_losses else float('nan')
    }, final_model_path)
    print(f"Final KVCompressor model saved to {final_model_path}")

    if all_step_losses: plot_losses(all_step_losses, save_path=os.path.join(output_dir, "losses_final.png"))
    
    # --- 4. 关闭 TensorBoard Writer ---
    writer.close()
    # ----------------------------------
    environment.close()
    print("Environment closed.")

if __name__ == "__main__":
    seed = TRAINING_PARAMS.get("RANDOM_SEED", 42) # 从TRAINING_PARAMS取
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Using random seed: {seed}")

    try:
        main_compressor_training_loop()
    except Exception as e:
        print(f"An error occurred during the KVCompressor training loop: {e}")
        import traceback
        traceback.print_exc()