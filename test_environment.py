from robotic_arm.environment import RoboticBaristaEnv
import time
import numpy as np

def main():
    # Create environment
    env = RoboticBaristaEnv(gui=True)
    
    # Reset to get initial observation
    observation = env.reset()
    print(f"Observation shape: {observation.shape}")
    
    # Wait a moment to view initial position
    time.sleep(1)
    
    # Test coffee preparation
    success = env.pour_from_container(0)  # 0 for coffee
    print(f"Coffee preparation success: {success}")
    
    # Reset environment
    env.reset()
    time.sleep(1)
    
    # Test matcha preparation
    success = env.pour_from_container(1)  # 1 for matcha
    print(f"Matcha preparation success: {success}")
    
    # Reset environment again
    env.reset()
    
    # Keep the window open to view the result
    print("\nTest complete. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()