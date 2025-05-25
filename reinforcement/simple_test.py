import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simple_rl_env import SimpleRLEnv
import pybullet as p

def test_agent(target_container="coffee", model_path=None):
    """Test a trained RL agent
    
    Args:
        target_container (str): "coffee" or "matcha"
        model_path (str): Path to trained model file
    """
    # Set default model path if not provided
    if model_path is None:
        # Try to load best model first, fall back to final model
        best_model_path = f"models/rl_models/{target_container}/best_model"
        final_model_path = f"models/rl_models/{target_container}_agent"
        
        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        else:
            model_path = final_model_path
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file {model_path}.zip not found.")
        print("Please train the model first using simple_train.py")
        return
    
    # Create environment with GUI
    env = SimpleRLEnv(target_container=target_container, gui=True)
    
    # Wrap for compatibility with stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Load normalization stats if available
    env_path = f"models/rl_models/{target_container}_env"
    if os.path.exists(env_path + ".pkl"):
        env = VecNormalize.load(env_path, env)
        # Don't update normalization stats during testing
        env.training = False
        env.norm_reward = False
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Reset environment
    obs = env.reset()
    
    # Run episode
    done = False
    total_reward = 0
    steps = 0
    marker_id = None
    
    print(f"Testing {target_container} preparation...")
    
    try:
        while not done and steps < 2000:  # Increased max steps for testing
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Unwrap info from VecEnv
            container_distance = info[0].get('container_distance', float('inf'))
            cup_distance = info[0].get('cup_distance', float('inf'))
            reached_container = info[0].get('reached_container', False)
            
            # Remove previous marker if it exists
            if marker_id is not None:
                p.removeBody(marker_id)
            
            # Add a visual marker at the current target
            if not reached_container:
                # Phase 1: Target is container
                target_pos = env.envs[0].target_position
                marker_color = [1, 0, 0, 0.7]  # Red for container
            else:
                # Phase 2: Target is cup
                target_pos = env.envs[0].cup_position
                marker_color = [0, 1, 0, 0.7]  # Green for cup
            
            # Create a small visual marker at the target position
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=0.05, 
                rgbaColor=marker_color
            )
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=target_pos
            )
            
            total_reward += reward[0]
            steps += 1
            
            # Slow down for better visualization
            time.sleep(0.01)
            
            # Print info occasionally
            if steps % 50 == 0:
                phase = "Cup" if reached_container else "Container"
                print(f"Step {steps}, Phase: {phase}, Reward: {reward[0]:.2f}, Total: {total_reward:.2f}")
                print(f"  Container distance: {container_distance:.3f}, Cup distance: {cup_distance:.3f}")
        
        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        
        # Keep environment open for viewing
        print("Press Ctrl+C to exit")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Close environment
        env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RL agent")
    parser.add_argument("--drink", type=str, choices=["coffee", "matcha"], default="coffee",
                       help="Drink type to test")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file (without .zip extension)")
    
    args = parser.parse_args()
    
    test_agent(target_container=args.drink, model_path=args.model)