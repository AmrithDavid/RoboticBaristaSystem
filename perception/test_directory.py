import os
import argparse
from test_classifier import test_classifier

def test_directory(model_path, directory, show_images=True):
    """Test all images in a directory
    
    Args:
        model_path (str): Path to the trained model
        directory (str): Directory containing images
        show_images (bool): Whether to display the images with results
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(directory, filename)
            image_paths.append(image_path)
    
    if not image_paths:
        print(f"No images found in directory: {directory}")
        return
    
    print(f"Found {len(image_paths)} images:")
    for path in image_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Test all found images
    test_classifier(model_path, image_paths, show_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test classifier on all images in a directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--directory", type=str, required=True,
                        help="Directory containing images for testing")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable image display")
    
    args = parser.parse_args()
    
    test_directory(args.model, args.directory, not args.no_display)