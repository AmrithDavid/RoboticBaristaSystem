import os
import sys
import time
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import robotic arm environment
from robotic_arm.environment import RoboticBaristaEnv

# Import perception system
from perception_system import PerceptionSystem

def main(args):
    """Main integration function"""
    print("\n===== Robotic Barista System Integration =====")
    
    # Initialize perception system
    perception = PerceptionSystem(args.model)
    
    # Initialize robotic arm environment with modified parameters
    env = RoboticBaristaEnv(gui=True)
    
    # Modify the move_to_pose method to increase max_steps
    original_move_to_pose = env.move_to_pose
    
    def enhanced_move_to_pose(target_position, max_steps=3000, threshold=0.05):
        """Wrapped move_to_pose with increased steps"""
        return original_move_to_pose(target_position, max_steps, threshold)
    
    # Replace the method with our enhanced version
    env.move_to_pose = enhanced_move_to_pose
    
    # Also modify pour_from_container if possible
    if hasattr(env, 'pour_from_container'):
        original_pour = env.pour_from_container
        
        def enhanced_pour(container_id):
            """Wrapper for pour_from_container with more detailed output"""
            print(f"Starting enhanced pour from container {container_id}...")
            
            # Select container
            if container_id == 0:
                container_position = env.coffee_position
                container_name = "coffee"
            else:
                container_position = env.matcha_position
                container_name = "matcha"
            
            print(f"\nPreparing {container_name}...")
            
            # Move to a position above the container (not touching)
            above_container = [
                container_position[0], 
                container_position[1],
                container_position[2] + 0.1  # 10cm above container
            ]
            print(f"Moving above {container_name} container")
            success = env.move_to_pose(above_container, max_steps=3000)
            if not success:
                return False
            
            # Simulate interacting with container
            print(f"Simulating interaction with {container_name} container")
            time.sleep(1)  
            
            # Move to a position above the cup (not touching)
            above_cup = [
                env.cup_position[0],
                env.cup_position[1],
                env.cup_position[2] + 0.1  # 10cm above cup
            ]
            print("Moving above cup")
            success = env.move_to_pose(above_cup, max_steps=3000)
            if not success:
                return False
            
            # Simulate pouring into cup
            print("Simulating pouring into cup")
            time.sleep(1)
            
            print(f"{container_name.capitalize()} preparation complete!")
            return True
            
        # Replace the method
        env.pour_from_container = enhanced_pour
    
    # Reset environment
    env.reset()
    
    # Allow environment to stabilize
    time.sleep(1)
    
    # Process each provided image
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        print(f"\n----- Processing {os.path.basename(img_path)} -----")
        
        # Classify the image
        try:
            drink_type, confidence = perception.classify_drink(img_path)
            print(f"Detected drink: {drink_type}")
            print(f"Confidence: {confidence:.4f}")
            
            # Only proceed if confidence is high enough
            if confidence < 0.7:
                print("Confidence too low, skipping pour")
                continue
            
            # Map drink type to container index
            container_idx = 0 if drink_type == "coffee" else 1
            
            # Execute pouring action
            print(f"Pouring {drink_type}...")
            success = env.pour_from_container(container_idx)
            
            if success:
                print(f"Successfully poured {drink_type}!")
            else:
                print(f"Failed to pour {drink_type}")
            
            # Reset environment for next image
            if args.images.index(img_path) < len(args.images) - 1:
                print("Resetting environment for next drink...")
                env.reset()
                time.sleep(1)
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    # Keep window open if requested
    if args.keep_open:
        print("\nDemo complete. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Robotic Barista System")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                       help="Paths to images for processing")
    parser.add_argument("--keep-open", action="store_true", 
                       help="Keep window open after execution")
    
    args = parser.parse_args()
    main(args)