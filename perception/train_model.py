import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def train_model(data_dir, output_dir, batch_size=32, num_epochs=20, learning_rate=0.001):
    """
    Train a CNN classifier on the dataset
    
    Args:
        data_dir (str): Directory containing 'train' and 'test' folders with class subfolders
        output_dir (str): Directory to save the trained model and plots
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load datasets
    image_datasets = {
        phase: datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
        for phase in ['train', 'test']
    }
    
    dataloaders = {
        phase: DataLoader(image_datasets[phase], batch_size=batch_size, 
                         shuffle=True, num_workers=0)
        for phase in ['train', 'test']
    }
    
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes: {class_names}")
    print(f"Training images: {dataset_sizes['train']}")
    print(f"Testing images: {dataset_sizes['test']}")
    
    # Save class names
    class_mapping = {i: class_name for i, class_name in enumerate(class_names)}
    
    # Load a pre-trained model
    model = models.resnet18(pretrained=True)
    
    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    training_stats = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store stats
            training_stats[f'{phase}_loss'].append(epoch_loss)
            training_stats[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save the best model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f'New best model saved with accuracy: {best_acc:.4f}')
        
        print()
    
    # Save the final model
    torch.save(model, os.path.join(output_dir, 'final_model.pth'))
    print(f'Final model saved. Best validation accuracy: {best_acc:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(training_stats['train_acc'], label='Train')
    plt.plot(training_stats['test_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(training_stats['train_loss'], label='Train')
    plt.plot(training_stats['test_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Print final message
    print('Training complete!')
    print(f'Best validation accuracy: {best_acc:.4f}')
    print(f'Models saved to {output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN classifier")
    parser.add_argument("--data-dir", type=str, default="../data",
                        help="Path to data directory containing 'train' and 'test' folders")
    parser.add_argument("--output-dir", type=str, default="../models",
                        help="Directory to save the trained model")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    
    args = parser.parse_args()
    
    train_model(
        args.data_dir,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.learning_rate
    )