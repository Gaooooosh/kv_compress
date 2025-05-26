# train_script.py
from typing import List
import torch
import torch.optim as optim
import numpy as np
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt

# 导入配置和自定义模块
from config_rl import TRAINING_PARAMS, LLM_CONFIG, DEFAULT_COMPRESSOR_CONFIG_PARAMS
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
    """主循环，用于训练KVCompressor"""
    print("Starting KVCompressor Training Loop...")
    start_time = datetime.now()

    # 1. 初始化设备
    device = torch.device(LLM_CONFIG["device"])
    print(f"Using device: {device}")

    # 2. 初始化 CompressorConfig
    #    RLEnvironment 初始化时会根据LLM配置覆盖部分参数
    compressor_cfg = CompressorConfig(
        input_dim=TRAINING_PARAMS.get("PLACEHOLDER_COMPRESSOR_INPUT_DIM", 1024),
        reduction_factor=DEFAULT_COMPRESSOR_CONFIG_PARAMS["reduction_factor"],
        output_seq_len=DEFAULT_COMPRESSOR_CONFIG_PARAMS["output_seq_len"],
        num_attention_heads=DEFAULT_COMPRESSOR_CONFIG_PARAMS["num_attention_heads"],
        use_mixed_precision=DEFAULT_COMPRESSOR_CONFIG_PARAMS["use_mixed_precision"],
        torch_dtype=DEFAULT_COMPRESSOR_CONFIG_PARAMS["torch_dtype"], # String, will be converted
        kv_head_dim=TRAINING_PARAMS.get("PLACEHOLDER_COMPRESSOR_KV_HEAD_DIM", 8),
        layer_nums=TRAINING_PARAMS.get("PLACEHOLDER_COMPRESSOR_LAYER_NUMS", 12),
        kernel_size=DEFAULT_COMPRESSOR_CONFIG_PARAMS["kernel_size"],
        padding=DEFAULT_COMPRESSOR_CONFIG_PARAMS["padding"],
        compression_threshold=DEFAULT_COMPRESSOR_CONFIG_PARAMS["compression_threshold"],
        initial_uncompressed_keep_length=DEFAULT_COMPRESSOR_CONFIG_PARAMS["initial_uncompressed_keep_length"]
    )
    print("Initial CompressorConfig created.")

    # 3. 初始化 CompressionEnv
    try:
        environment = CompressionEnv(
            compressor_config=compressor_cfg, # compressor_cfg会被env内部根据LLM调整
            llm_model_name_or_path=LLM_CONFIG["model_name_or_path"],
            tokenizer_name_or_path=LLM_CONFIG["tokenizer_name_or_path"],
            llm_device=LLM_CONFIG["device"],
            training_params=TRAINING_PARAMS,
            llm_config_params=LLM_CONFIG
        )
    except Exception as e:
        print(f"FATAL: Error initializing CompressionEnv: {e}")
        import traceback
        traceback.print_exc()
        return

    print("CompressionEnv initialized.")

    # 4. 初始化 KVCompressor 的优化器
    #    参数来自 environment.comp_cache 中的 k_compressor 和 v_compressor
    params_to_optimize = list(environment.comp_cache.k_compressor.parameters()) + \
                         list(environment.comp_cache.v_compressor.parameters())
    
    # 确保参数需要梯度
    num_trainable_params = 0
    for p in params_to_optimize:
        p.requires_grad = True # 显式设置
        if p.requires_grad:
            num_trainable_params += p.numel()

    if num_trainable_params == 0:
        print("FATAL: No trainable parameters found in KVCompressors. Check model definition and requires_grad settings.")
        return
    print(f"KVCompressor has {num_trainable_params} trainable parameters.")

    compressor_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params_to_optimize), 
        lr=TRAINING_PARAMS["COMPRESSOR_LEARNING_RATE"]
    )
    print(f"KVCompressor optimizer initialized with LR: {TRAINING_PARAMS['COMPRESSOR_LEARNING_RATE']}.")

    # (可选) 加载预训练的Compressor模型 (如果适用)
    load_compressor_path = TRAINING_PARAMS.get("LOAD_PRETRAINED_COMPRESSOR_PATH", None)
    if load_compressor_path and os.path.exists(load_compressor_path):
        print(f"Loading pretrained KVCompressor weights from: {load_compressor_path}")
        # 需要一种方式来加载k_compressor和v_compressor的权重
        # 假设保存的是一个包含 'k_compressor_state_dict' 和 'v_compressor_state_dict' 的字典
        try:
            checkpoint = torch.load(load_compressor_path, map_location=device)
            if 'k_compressor_state_dict' in checkpoint:
                environment.comp_cache.k_compressor.load_state_dict(checkpoint['k_compressor_state_dict'])
            if 'v_compressor_state_dict' in checkpoint:
                environment.comp_cache.v_compressor.load_state_dict(checkpoint['v_compressor_state_dict'])
            # (可选) 加载优化器状态
            # if 'optimizer_state_dict' in checkpoint and load_optimizer_state:
            #     compressor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Pretrained KVCompressor weights loaded.")
        except Exception as e:
            print(f"Warning: Could not load pretrained compressor weights from {load_compressor_path}: {e}")


    # 5. 训练循环 (基于总步数)
    all_step_losses = []
    output_dir = "compressor_training_output"
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join(output_dir, "saved_compressor_models")
    os.makedirs(model_save_dir, exist_ok=True)

    environment.reset() # 初始化环境状态和对话历史

    print(f"\nStarting KVCompressor training for {TRAINING_PARAMS['NUM_TRAINING_STEPS']} steps...")
    for step in range(1, TRAINING_PARAMS["NUM_TRAINING_STEPS"] + 1):
        
        # 环境执行一步，生成LLM输出，并计算损失
        # generate_one_step 内部会处理梯度上下文，确保KVCompressor参数的梯度被追踪
        loss_tensor, done, info = environment.generate_one_step()

        if loss_tensor is None:
            print(f"Warning: Step {step} returned None for loss. Skipping update.")
            if done: # 如果是因为错误导致回合结束，也需要重置
                print(f"  Resetting environment due to error or early termination in step.")
                environment.reset()
            continue

        # 优化KVCompressor参数
        if loss_tensor.requires_grad: # 确保损失是可反向传播的
            compressor_optimizer.zero_grad()
            loss_tensor.backward()

            if TRAINING_PARAMS.get("GRADIENT_CLIP_NORM", 0) > 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, TRAINING_PARAMS["GRADIENT_CLIP_NORM"])
            
            compressor_optimizer.step()
        else:
            # print(f"Warning: Loss tensor at step {step} does not require grad. No optimization performed.")
            pass # 不报错，但记录下来

        all_step_losses.append(loss_tensor.item())

        if step % TRAINING_PARAMS["LOG_FREQ_STEPS"] == 0:
            avg_loss_recent = np.mean(all_step_losses[-100:]) if len(all_step_losses) > 0 else float('nan')
            print(f"Step {step}/{TRAINING_PARAMS['NUM_TRAINING_STEPS']}: "
                  f"Loss: {loss_tensor.item():.6f}, Avg Loss (last 100): {avg_loss_recent:.6f}, "
                  f"Info: compr_triggered={info.get('compression_triggered_this_step', False)}, "
                  f"compr_performed={info.get('compression_performed_this_step', False)}, "
                  f"new_tokens_main={info.get('newly_generated_token_main', -1)}")

        # 如果一个“回合”（数据段）结束，重置环境
        if done:
            print(f"  Segment/Episode finished at step {step} (total tokens: {environment._current_episode_total_tokens}). Resetting environment.")
            environment.reset()
        
        # 定期保存模型
        if step % TRAINING_PARAMS["COMPRESSOR_MODEL_SAVE_FREQ_STEPS"] == 0:
            model_save_path = os.path.join(model_save_dir, f"kv_compressor_step_{step}.pth")
            torch.save({
                'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(),
                'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(),
                'optimizer_state_dict': compressor_optimizer.state_dict(),
                'step': step,
                'loss': loss_tensor.item()
            }, model_save_path)
            print(f"KVCompressor model saved to {model_save_path}")
            
            # 绘制并保存损失曲线图
            if len(all_step_losses) > 10: # 至少有一些数据点再画图
                plot_losses(all_step_losses, save_path=os.path.join(output_dir, f"losses_step_{step}.png"))

    # 训练结束
    end_time = datetime.now()
    print(f"\nKVCompressor training finished in {end_time - start_time}.")
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "kv_compressor_final.pth")
    torch.save({
        'k_compressor_state_dict': environment.comp_cache.k_compressor.state_dict(),
        'v_compressor_state_dict': environment.comp_cache.v_compressor.state_dict(),
        'optimizer_state_dict': compressor_optimizer.state_dict(),
        'step': TRAINING_PARAMS['NUM_TRAINING_STEPS'],
        'final_avg_loss': np.mean(all_step_losses[-1000:]) if len(all_step_losses) > 0 else float('nan')
    }, final_model_path)
    print(f"Final KVCompressor model saved to {final_model_path}")

    # 保存最终的损失曲线
    if len(all_step_losses) > 0:
        plot_losses(all_step_losses, save_path=os.path.join(output_dir, "losses_final.png"))

    # 清理环境
    environment.close()
    print("Environment closed.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    seed = TRAINING_PARAMS.get("RANDOM_SEED", None) # 从TRAINING_PARAMS取
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