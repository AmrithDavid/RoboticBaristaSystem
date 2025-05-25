import os
import argparse
import torch
from perception_system import PerceptionSystem
import matplotlib.pyplot as plt
import cv2
import numpy as np

def test_classifier(model_path, image_paths, show_images=True):
    """Test the classifier on one or more images
    
    Args:
        model_path (str): Path to the trained model
        image_paths (list): List of image paths to classify
        show_images (bool): Whether to display the images with results
    """
    # Initialize perception system
    perception = PerceptionSystem(model_path)
    
    results = []
    
    # Process each image
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        print(f"\nTesting image: {os.path.basename(img_path)}")
        
        try:
            # Classify the image
            drink_type, confidence = perception.classify_drink(img_path)
            
            print(f"Predicted: {drink_type}")
            print(f"Confidence: {confidence:.4f}")
            
            # Display a clearer result
            if confidence > 0.7:
                result_text = f"This is {drink_type.upper()} with high confidence"
            else:
                result_text = f"This might be {drink_type} but confidence is low"
            
            print(f"RESULT: {result_text}")
            
            # Store result for display
            results.append({
                'path': img_path,
                'drink_type': drink_type,
                'confidence': confidence,
                'result_text': result_text
            })
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Display images with results
    if show_images and results:
        display_results(results)
    
    return results

def display_results(results):
    """Display images with classification results
    
    Args:
        results (list): List of result dictionaries
    """
    num_images = len(results)
    
    # Determine grid size
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    for i, result in enumerate(results):
        # Load and convert image
        img = cv2.imread(result['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create subplot
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        
        # Add title with result
        title = f"{result['drink_type'].upper()} ({result['confidence']:.2f})"
        plt.title(title, color='green' if result['confidence'] > 0.7 else 'red')
        
        # Add filename
        plt.xlabel(os.path.basename(result['path']))
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the drink classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                        help="Paths to images for testing")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable image display")
    
    args = parser.parse_args()
    
    test_classifier(args.model, args.images, not args.no_display)