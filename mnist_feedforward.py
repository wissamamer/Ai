#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Session 2: Guided Exercise - MNIST Digit Recognition with Feedforward Neural Networks
===================================================================================

This script implements a complete feedforward neural network from scratch
for classifying handwritten digits from the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define parameters
batch_size = 64
learning_rate = 0.01
num_epochs = 5
hidden_size = 128

# PART 1: DATA LOADING AND PREPROCESSING
# ======================================

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

# Download and load training data
train_dataset = datasets.MNIST(root='./data', 
                              train=True, 
                              download=True, 
                              transform=transform)

# Download and load test data
test_dataset = datasets.MNIST(root='./data', 
                             train=False, 
                             download=True, 
                             transform=transform)

# Create data loaders for batch processing
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# PART 2: VISUALIZE SOME SAMPLE IMAGES
# ===================================

def visualize_samples(data_loader, num_samples=10):
    """Visualize sample images from the dataset"""
    
    examples = iter(data_loader)
    samples, labels = next(examples)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(samples[i][0], cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.show()

# Uncomment to visualize samples
# visualize_samples(train_loader)


# PART 3: DEFINE THE NEURAL NETWORK ARCHITECTURE
# ============================================

class FeedforwardNeuralNet(nn.Module):
    """A simple feedforward neural network with one hidden layer"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the network architecture
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes
        """
        super(FeedforwardNeuralNet, self).__init__()
        
        # First fully connected layer (input → hidden)
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Second fully connected layer (hidden → output)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Dropout layer for regularization (preventing overfitting)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Reshape input: [batch_size, 1, 28, 28] → [batch_size, 784]
        x = x.view(-1, 28*28)
        
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer (no activation yet - will use softmax with loss function)
        x = self.fc2(x)
        
        return x


# PART 4: INITIALIZE THE MODEL, LOSS FUNCTION, AND OPTIMIZER
# =========================================================

# Initialize the model
input_size = 28 * 28  # MNIST images are 28x28 pixels
output_size = 10      # 10 digits (0-9)
model = FeedforwardNeuralNet(input_size, hidden_size, output_size)

# Define the loss function (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# PART 5: TRAINING LOOP
# ===================

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []

def train(model, train_loader, criterion, optimizer, epoch):
    """Train the model for one epoch
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        epoch: Current epoch number
    
    Returns:
        Average loss and accuracy for this epoch
    """
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Clear gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print statistics every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    
    # Calculate average metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # Store for plotting
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    return epoch_loss, epoch_acc


# PART 6: EVALUATION FUNCTION
# =========================

def evaluate(model, test_loader):
    """Evaluate model performance on the test set
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
        
    Returns:
        Test accuracy
    """
    # Set model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    
    # Disable gradient calculation
    with torch.no_grad():
        for data, target in test_loader:
            # Forward pass
            outputs = model(data)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate and return accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy


# PART 7: TRAIN THE MODEL
# =====================

def train_model():
    """Complete training process"""
    
    print("Starting training...")
    
    # Train for multiple epochs
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = train(model, train_loader, criterion, optimizer, epoch)
        print(f'Epoch {epoch+1}/{num_epochs} completed - '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    print("Training finished!")
    
    # Evaluate on test set
    test_accuracy = evaluate(model, test_loader)
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    return test_accuracy


# PART 8: VISUALIZE PREDICTIONS
# ===========================

def visualize_predictions(model, test_loader, num_samples=10):
    """Visualize model predictions on sample test images
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
        num_samples: Number of samples to visualize
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of test data
    examples = iter(test_loader)
    samples, labels = next(examples)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(samples)
        _, predicted = torch.max(outputs, 1)
    
    # Plot results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(samples[i][0], cmap='gray')
        
        # Green title for correct predictions, red for incorrect
        if predicted[i] == labels[i]:
            plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}', color='green')
        else:
            plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}', color='red')
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()


# PART 9: RUN THE COMPLETE EXERCISE
# ===============================

def main():
    """Execute the complete exercise"""
    
    # Visualize sample images
    visualize_samples(train_loader)
    
    # Train the model
    test_accuracy = train_model()
    
    # Visualize predictions
    visualize_predictions(model, test_loader)
    
    # Print final accuracy
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_feedforward_model.pth')
    print("Model saved successfully!")


if __name__ == "__main__":
    main()

# PART 10: EXTENSIONS AND CHALLENGES
# ================================

"""
Suggested extensions to improve the model:
1. Add more hidden layers to create a deeper network
2. Experiment with different activation functions (Sigmoid, Tanh, Leaky ReLU)
3. Implement learning rate scheduling 
4. Try different optimizers (Adam, RMSprop)
5. Add batch normalization for faster and more stable training
6. Implement early stopping to prevent overfitting
7. Use data augmentation to improve generalization
8. Convert to a CNN architecture for better performance
"""

# Example extension: Create a deeper network
class DeepFeedforwardNet(nn.Module):
    """A deeper feedforward neural network with multiple hidden layers"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        """Initialize the network architecture
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of sizes for each hidden layer
            output_size: Number of output classes
        """
        super(DeepFeedforwardNet, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = x.view(-1, 28*28)  # Flatten the input
        return self.model(x)

# Uncomment to use the deeper model
"""
deeper_model = DeepFeedforwardNet(
    input_size=28*28,
    hidden_sizes=[256, 128, 64],
    output_size=10
)
"""
