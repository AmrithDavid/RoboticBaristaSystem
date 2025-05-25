import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

class PerceptionSystem:
    """Computer vision system for drink classification using PyTorch"""
    
    def __init__(self, model_path):
        """Initialize the perception system
        
        Args:
            model_path (str): Path to the trained classifier model
        """
        # Load the model safely
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a ResNet model with the same architecture as our trained model
        base_model = models.resnet18(pretrained=False)
        num_classes = 2  # coffee and matcha
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, num_classes)
        
        try:
            # Try loading the state dict instead of the full model
            state_dict = torch.load(model_path, map_location=self.device)
            
            # If state_dict is already a state dict, use it; otherwise try to access the state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                base_model.load_state_dict(state_dict['state_dict'])
            else:
                base_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model state dict, trying unsafe load: {e}")
            # Try with unsafe load as fallback
            try:
                # Add specific allowed class for safe loading
                torch.serialization.add_safe_globals(['torchvision.models.resnet.ResNet'])
                self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                print("Model loaded using unsafe method")
            except Exception as inner_e:
                print(f"Failed to load model: {inner_e}")
                raise
        else:
            # If state dict load succeeded
            self.model = base_model
        
        self.model.eval()  # Set to evaluation mode
        print(f"Loaded classifier from {model_path}")
        
        # Class labels
        self.class_labels = ["coffee", "matcha"]
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, img_path):
        """Preprocess image for the model
        
        Args:
            img_path: Path to image file
        
        Returns:
            Preprocessed image tensor ready for prediction
        """
        # Open image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def classify_drink(self, img_path):
        """Classify the drink type from an image
        
        Args:
            img_path: Path to image file
        
        Returns:
            tuple: (class_label, confidence)
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(img_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
        class_label = self.class_labels[predicted_class_idx]
        
        return class_label, confidence
    
    def process_image_file(self, image_path):
        """Process an image file
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            dict: Results containing drink type and confidence
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Classify the drink
        drink_type, confidence = self.classify_drink(image_path)
        
        # Load image for visualization (optional)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            "image": img,
            "drink_type": drink_type,
            "confidence": float(confidence)
        }

# Simple test if run directly
if __name__ == "__main__":
    # This will run if you execute this file directly
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        # Initialize perception system
        perception = PerceptionSystem(model_path)
        
        # Test with some images if provided
        for i, img_path in enumerate(sys.argv[2:]):
            if os.path.exists(img_path):
                print(f"\nTesting image {i+1}: {img_path}")
                try:
                    drink_type, confidence = perception.classify_drink(img_path)
                    print(f"Predicted: {drink_type} with confidence {confidence:.2f}")
                    
                    # Display result with more clarity
                    if confidence > 0.7:
                        print(f"RESULT: This is {drink_type.upper()} with high confidence")
                    else:
                        print(f"RESULT: This might be {drink_type} but confidence is low")
                except Exception as e:
                    print(f"Error processing image: {e}")