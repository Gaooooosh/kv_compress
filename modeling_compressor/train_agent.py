# train_agent.py
from collections import deque
import random
from typing import List
import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt # 用于绘制学习曲线

# 导入配置和自定义模块
from config_rl import RL_PARAMS, LLM_CONFIG
from untis import CompressorConfig # 假设untis.py在同级目录且包含CompressorConfig
from CCache import CompCache
from rl_environment import RLEnvironment
from rl_agent import DQNAgent

def plot_rewards(episode_rewards: List[float], episode_avg_rewards: List[float], save_path: str = "training_rewards.png"):
    """绘制每回合奖励和平均奖励曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label="Reward per Episode", alpha=0.7)
    plt.plot(episode_avg_rewards, label="Average Reward (100 episodes)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to {save_path}")

def main_training_loop():
    """主训练循环"""
    print("Starting Main Training Loop...")
    start_time = datetime.now()

    # 1. 初始化设备
    device = torch.device(LLM_CONFIG["device"])
    print(f"Using device: {device}")

    # 2. 初始化 CompressorConfig
    #    这里的参数可以从 RL_PARAMS 或专门的 compressor 配置中读取
    #    注意：RLEnvironment 初始化时会根据LLM配置覆盖部分 CompressorConfig 参数
    #    (如 layer_nums, kv_head_dim, input_dim)
    #    所以这里的初始值可以是占位符或合理的默认值。
    compressor_cfg = CompressorConfig(
        input_dim=RL_PARAMS.get("PLACEHOLDER_COMPRESSOR_INPUT_DIM", 1024), # 会被Env覆盖
        reduction_factor=RL_PARAMS.get("DEFAULT_COMPRESSOR_REDUCTION_FACTOR", 4),
        output_seq_len=RL_PARAMS.get("DEFAULT_COMPRESSOR_OUTPUT_SEQ_LEN", 8),
        num_attention_heads=RL_PARAMS.get("DEFAULT_COMPRESSOR_NUM_ATTN_HEADS", 8),
        use_mixed_precision=RL_PARAMS.get("COMPRESSOR_USE_MIXED_PRECISION", False),
        torch_dtype=torch.float32 if not RL_PARAMS.get("COMPRESSOR_USE_MIXED_PRECISION", False) else torch.float16,
        kv_head_dim=RL_PARAMS.get("PLACEHOLDER_COMPRESSOR_KV_HEAD_DIM", 8), # 会被Env覆盖
        layer_nums=RL_PARAMS.get("PLACEHOLDER_COMPRESSOR_LAYER_NUMS", 12), # 会被Env覆盖
        kernel_size=RL_PARAMS.get("DEFAULT_COMPRESSOR_KERNEL_SIZE", 3),
        padding=RL_PARAMS.get("DEFAULT_COMPRESSOR_PADDING", 1),
        compression_threshold=RL_PARAMS.get("DEFAULT_COMPRESSION_THRESHOLD", 64),
        initial_uncompressed_keep_length=RL_PARAMS.get("DEFAULT_INITIAL_UNCOMPRESSED_KEEP_LENGTH", 32)
    )
    print("CompressorConfig initialized (some params may be overridden by RLEnvironment).")

    # 3. 初始化 RLEnvironment
    #    RLEnvironment 内部会加载LLM和Tokenizer，并初始化CompCache
    try:
        environment = RLEnvironment(
            compressor_config=compressor_cfg, # compressor_cfg会被env内部根据LLM调整
            llm_model_name_or_path=LLM_CONFIG["model_name_or_path"],
            tokenizer_name_or_path=LLM_CONFIG["tokenizer_name_or_path"],
            llm_device=LLM_CONFIG["device"],
            rl_params=RL_PARAMS,
            llm_config_params=LLM_CONFIG
        )
    except Exception as e:
        print(f"FATAL: Error initializing RLEnvironment: {e}")
        print("Please check your configurations, model paths, and dependencies.")
        return # 无法继续

    print("RLEnvironment initialized.")

    # 4. 初始化 DQNAgent
    state_dim = environment.state_dim
    action_dim = environment.action_dim
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        rl_config=RL_PARAMS # 传入完整的RL_PARAMS字典
    )
    print(f"DQNAgent initialized with state_dim={state_dim}, action_dim={action_dim}.")

    # (可选) 加载预训练的Agent模型
    load_pretrained_agent_path = RL_PARAMS.get("LOAD_PRETRAINED_AGENT_PATH", None)
    if load_pretrained_agent_path and os.path.exists(load_pretrained_agent_path):
        print(f"Loading pretrained agent model from: {load_pretrained_agent_path}")
        agent.load_model(load_pretrained_agent_path)
    else:
        print("No pretrained agent model loaded, starting training from scratch.")


    # 5. 训练循环
    all_episode_rewards = []
    avg_rewards_history = [] # 用于存储每100回合的平均奖励
    recent_episode_rewards = deque(maxlen=100) # 用于计算滑动平均奖励
    global_step_counter = 0 # 用于epsilon衰减和学习频率控制

    output_dir = "training_output"
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"\nStarting training for {RL_PARAMS['NUM_EPISODES']} episodes...")
    for episode in range(1, RL_PARAMS["NUM_EPISODES"] + 1):
        current_state = environment.reset() # 环境返回初始状态 (np.ndarray)
        episode_reward = 0
        episode_loss_sum = 0
        episode_learn_counts = 0

        for step in range(1, RL_PARAMS["MAX_STEPS_PER_EPISODE"] + 1):
            # Agent选择动作
            action = agent.select_action(current_state, is_training=True)

            # 环境执行动作
            next_state, reward, done, info = environment.step(action)

            # Agent存储经验
            agent.store_experience(current_state, action, reward, next_state, done)

            current_state = next_state
            episode_reward += reward
            global_step_counter += 1

            # Agent学习 (如果满足条件)
            if len(agent.replay_buffer) >= RL_PARAMS["MIN_EXPERIENCES_FOR_LEARNING"] and \
               global_step_counter % RL_PARAMS["AGENT_LEARN_FREQ"] == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_loss_sum += loss
                    episode_learn_counts += 1
            
            if RL_PARAMS["LOG_FREQ_STEPS"] > 0 and step % RL_PARAMS["LOG_FREQ_STEPS"] == 0:
                print(f"  Episode {episode}/{RL_PARAMS['NUM_EPISODES']}, Step {step}/{RL_PARAMS['MAX_STEPS_PER_EPISODE']}: "
                      f"Action: {'Compress' if action == 1 else 'NoComp'}, "
                      f"Reward: {reward:.3f}, Done: {done}, "
                      f"CacheTotal: {info.get('current_cache_total_len', 'N/A')}, "
                      f"Uncomp: {info.get('current_cache_uncompressed_len', 'N/A')}, "
                      f"Comp: {info.get('current_cache_compressed_len', 'N/A')}, "
                      f"Epsilon: {agent.current_epsilon:.4f}")


            if done:
                break
        
        all_episode_rewards.append(episode_reward)
        recent_episode_rewards.append(episode_reward)
        avg_reward_last_100 = sum(recent_episode_rewards) / len(recent_episode_rewards)
        avg_rewards_history.append(avg_reward_last_100)
        avg_loss_this_episode = (episode_loss_sum / episode_learn_counts) if episode_learn_counts > 0 else 0.0

        print(f"Episode {episode} finished after {step} steps. "
              f"Total Reward: {episode_reward:.2f}, "
              f"Avg Reward (last 100): {avg_reward_last_100:.2f}, "
              f"Avg Loss: {avg_loss_this_episode:.4f}, "
              f"Epsilon: {agent.current_epsilon:.4f}, "
              f"Buffer Size: {len(agent.replay_buffer)}")

        # 定期保存模型
        if episode % RL_PARAMS["MODEL_SAVE_FREQ_EPISODES"] == 0:
            model_save_path = os.path.join(model_save_dir, f"dqn_agent_episode_{episode}.pth")
            agent.save_model(model_save_path)
            # 绘制并保存奖励曲线图
            plot_rewards(all_episode_rewards, avg_rewards_history, save_path=os.path.join(output_dir, f"rewards_episode_{episode}.png"))


    # 训练结束
    end_time = datetime.now()
    print(f"\nTraining finished in {end_time - start_time}.")
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "dqn_agent_final.pth")
    agent.save_model(final_model_path)
    print(f"Final agent model saved to {final_model_path}")

    # 保存最终的奖励曲线
    plot_rewards(all_episode_rewards, avg_rewards_history, save_path=os.path.join(output_dir, "rewards_final.png"))

    # 清理环境
    environment.close()
    print("Environment closed.")

if __name__ == "__main__":
    # 设置随机种子以保证可复现性 (可选)
    seed = RL_PARAMS.get("RANDOM_SEED", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Using random seed: {seed}")

    try:
        main_training_loop()
    except Exception as e:
        print(f"An error occurred during the training loop: {e}")
        import traceback
        traceback.print_exc()