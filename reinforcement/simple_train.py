import os
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from simple_rl_env import SimpleRLEnv

class EnhancedCallback(BaseCallback):
    """
    Callback for saving a model and advancing curriculum when reward improves
    """
    def __init__(self, check_freq=1000, save_path="models/rl_models", verbose=1):
        super(EnhancedCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        self.success_counter = 0
        self.success_threshold = 5  # Number of successes needed to advance
    
    def _init_callback(self) -> None:
        # Create folder if needed
        os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Compute mean reward over last 100 episodes
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else -float('inf')
            
            # Check for successes (episodes with high rewards)
            successes = sum(1 for ep_info in self.model.ep_info_buffer if ep_info["r"] > 50) if len(self.model.ep_info_buffer) > 0 else 0
            
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Mean reward: {mean_reward:.2f}, Successes: {successes}")
            
            # Save model if mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
                if self.verbose > 0:
                    print(f"Saving model to {self.save_path}/best_model")
            
            # Check if we should advance curriculum
            if successes >= self.success_threshold:
                # Reset counter
                self.success_counter = 0
                
                # Advance curriculum in all environments
                for env in self.model.env.envs:
                    # Access the actual environment inside the wrapper
                    if hasattr(env, 'advance_curriculum'):
                        env.advance_curriculum()
                    elif hasattr(env, 'env') and hasattr(env.env, 'advance_curriculum'):
                        env.env.advance_curriculum()
                    elif hasattr(env, 'venv') and hasattr(env.venv.envs[0], 'advance_curriculum'):
                        for subenv in env.venv.envs:
                            subenv.advance_curriculum()
                
                if self.verbose > 0:
                    print("Curriculum advanced!")
        
        return True

def train_simple_agent(target_container="coffee", timesteps=300000, gui=False):
    """Train a simple RL agent for the robotic barista
    
    Args:
        target_container (str): "coffee" or "matcha"
        timesteps (int): Number of training timesteps
        gui (bool): Whether to show GUI during training
    """
    print(f"Training agent for {target_container}...")
    
    # Create environment with curriculum learning (start with easy phase 0)
    def make_env():
        return SimpleRLEnv(target_container=target_container, gui=gui, curriculum_phase=0)
    
    # Wrap in DummyVecEnv for stable-baselines
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment (with hard difficulty for evaluation)
    def make_eval_env():
        return SimpleRLEnv(target_container=target_container, gui=False, curriculum_phase=2)
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create PPO agent with improved parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Increased exploration
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
        verbose=1
    )
    
    # Setup callback
    callback = EnhancedCallback(check_freq=5000, save_path=f"models/rl_models/{target_container}")
    
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    training_time = time.time() - start_time
    
    # Save the final model
    model_path = f"models/rl_models/{target_container}_agent"
    model.save(model_path)
    
    # Save the environment normalization statistics
    env_path = f"models/rl_models/{target_container}_env"
    env.save(env_path)
    
    print(f"Model saved to {model_path}")
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train simple RL agent")
    parser.add_argument("--drink", type=str, choices=["coffee", "matcha"], default="coffee",
                       help="Drink type to train for")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Number of timesteps to train")
    parser.add_argument("--gui", action="store_true",
                       help="Show GUI during training (slower)")
    
    args = parser.parse_args()
    
    train_simple_agent(
        target_container=args.drink,
        timesteps=args.timesteps,
        gui=args.gui
    )