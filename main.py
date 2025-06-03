import os
import random
import argparse
import subprocess
import sys

def main(args):
    # Validate inputs
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found - {args.model}")
        sys.exit(1)
        
    if not os.path.isdir(args.images_dir):
        print(f"Error: Image directory not found - {args.images_dir}")
        sys.exit(1)
    
    # Collect valid images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [
        os.path.join(args.images_dir, f) 
        for f in os.listdir(args.images_dir) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not all_images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)
    
    # Randomly select images
    num_select = min(args.orders, len(all_images))
    selected_images = random.sample(all_images, num_select)
    
    print("\n=== Randomly Selected Orders ===")
    for i, img in enumerate(selected_images):
        print(f"{i+1}. {os.path.basename(img)}")
    
    # Build integrate.py command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "perception\integrate.py",
        "--model", args.model,
        "--images"
    ]
    cmd.extend(selected_images)
    if args.keep_open:
        cmd.append("--keep-open")
    
    
    print("\n===== Executing Robotic Barista System Integration Command=====")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Execute integrate.py
    try:
        subprocess.run(cmd, check=True)
        print("\n=== Order Processing Complete ===")
    except subprocess.CalledProcessError as e:
        print(f"\nError: System execution failed (code {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Order Processor for Robotic Barista System')
    parser.add_argument('--model', type=str, default="models/final_model.pth",
                       help='Path to trained perception model')
    parser.add_argument('--images-dir', type=str, default="data/true_test",
                       help='Directory containing drink images')
    parser.add_argument('--orders', type=int, default=2,
                       help='Number of random orders to process (default: 2)')
    parser.add_argument('--keep-open', action='store_true',
                       help='Keep simulation window open after processing')
    
    args = parser.parse_args()
    main(args)
