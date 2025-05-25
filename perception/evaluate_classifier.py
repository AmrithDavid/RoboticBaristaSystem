import os
import numpy as np
import torch
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from perception_system import PerceptionSystem
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(model_path, test_dir, show_plots=True):
    """Evaluate the classifier's performance metrics
    
    Args:
        model_path (str): Path to the trained model
        test_dir (str): Directory containing test images
        show_plots (bool): Whether to display plots interactively
    """
    print(f"Evaluating model: {model_path}")
    print(f"Test directory: {test_dir}")
    
    # Initialize perception system
    perception = PerceptionSystem(model_path)
    
    # Lists to store true labels and predictions
    y_true = []
    y_pred = []
    confidences = []
    
    # Class mapping
    class_to_idx = {"coffee": 0, "matcha": 1}
    idx_to_class = {0: "coffee", 1: "matcha"}
    
    # Process each class folder
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {class_name}")
        
        # Check if class is in our mapping
        if class_name not in class_to_idx:
            print(f"Warning: Unknown class {class_name}, skipping")
            continue
        
        # True label for this class
        true_label = class_to_idx[class_name]
        
        # Process each image in the class folder
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, filename)
            
            # Classify the image
            try:
                pred_class, confidence = perception.classify_drink(img_path)
                pred_label = class_to_idx[pred_class]
                
                # Store results
                y_true.append(true_label)
                y_pred.append(pred_label)
                confidences.append(confidence)
                
                # Output for each image
                result = "✓" if pred_label == true_label else "✗"
                print(f"  {filename}: Predicted {pred_class} ({confidence:.4f}) {result}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    # Calculate metrics
    if len(y_true) == 0:
        print("No images were processed. Check your test directory.")
        return
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Calculate overall metrics
    overall_precision = precision_score(y_true, y_pred, average='weighted')
    overall_recall = recall_score(y_true, y_pred, average='weighted')
    overall_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Total images: {len(y_true)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nPer-class Metrics:")
    for i, class_name in idx_to_class.items():
        print(f"  {class_name}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1 Score: {f1[i]:.4f}")
    
    print("\nOverall Metrics:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")
    
    print("\nConfidence Statistics:")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(idx_to_class.values()),
                yticklabels=list(idx_to_class.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Plot confidence distribution
    plt.figure(figsize=(8, 6))
    for i, class_name in idx_to_class.items():
        class_confidences = confidences[y_true == i]
        if len(class_confidences) > 0:
            plt.hist(class_confidences, alpha=0.5, label=class_name, bins=10)
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png')
    
    print("\nPlots saved to 'confusion_matrix.png' and 'confidence_distribution.png'")
    
    if show_plots:
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier performance")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--test-dir", type=str, required=True,
                       help="Directory containing test images")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable showing plots")
    
    args = parser.parse_args()
    
    evaluate_classifier(args.model, args.test_dir, not args.no_plots)