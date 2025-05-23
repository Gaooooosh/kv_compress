import torch
import numpy as np

class TrajectoryBuffer:
    def __init__(self, buffer_size, num_llm_batch_size, num_segments_per_step, state_segment_shape, action_segment_shape, device):
        self.buffer_size = buffer_size # Max number of (state,action,...) transitions for segments
        self.num_llm_batch_size = num_llm_batch_size
        self.num_segments_per_step = num_segments_per_step # Max layers compressed at once
        self.device = device

        # Assuming state_segment_shape is (S_segment, D_orig_feat)
        # Assuming action_segment_shape is (S_compressed, D_orig_feat)
        # These will be flattened or handled. Let's store raw segments for now.
        # This buffer will store experiences from *one iteration* of rollouts.
        self.states = [] # List of state tensors for segments
        self.actions = [] # List of action tensors for segments
        self.log_probs_old = [] # List of log_probs
        self.rewards = [] # List of step rewards (scalar)
        self.values = [] # List of value estimates (scalar)
        self.dones = [] # List of done flags (boolean/float) - for end of LLM generation episode

        self.ptr = 0
        self.path_start_idx = 0

    def add_experience(self, states_batch, actions_batch, log_probs_batch, rewards_step, values_batch, dones_step):
        """
        Adds experiences from one compression step.
        states_batch: (B_llm, num_segments, S_seg, D_orig) -> to be processed by actor/critic
        actions_batch: (B_llm, num_segments, S_comp, D_comp) -> output of actor
        log_probs_batch: (B_llm, num_segments)
        rewards_step: (B_llm,) - reward for this LLM generation step, attributed to these compressions
        values_batch: (B_llm, num_segments)
        dones_step: (B_llm,) - if LLM episode ended after this step
        """
        # This needs careful handling of batching if B_llm > 1 or num_segments > 1
        # For simplicity, assume we flatten these or store them directly
        # For now, let's assume inputs are already appropriately batched/processed
        # and we are storing flat lists of transitions.

        # Example: if we process one LLM batch item and one segment at a time for storage
        # This part needs detailed design based on how PPO expects data.
        # For now, let's just append.
        self.states.append(states_batch) # This will be a list of batches of segments
        self.actions.append(actions_batch)
        self.log_probs_old.append(log_probs_batch)
        self.rewards.append(rewards_step) # This needs to be distributed correctly if multiple segments
        self.values.append(values_batch)
        self.dones.append(dones_step) # Also needs to be distributed

    def get_trajectories(self):
        # This method would process the stored lists into tensors for PPO updates
        # and calculate advantages/returns.
        # For simplicity, PPOAgent.update will take this buffer and do it.
        # Data needs to be [s1,s2,...,sN], [a1,a2,...,aN], etc.
        # Current storage is [[s_batch1_seg1..k], [s_batch2_seg1..k], ...]
        # This requires careful unrolling.
        pass

    def clear(self):
        self.states, self.actions, self.log_probs_old, self.rewards, self.values, self.dones = [], [], [], [], [], []