# PyTorch Tensor Operations - Practical Exercise
# Session 1: Introduction to PyTorch Tensors and Basic Operations

import torch
import matplotlib.pyplot as plt
import numpy as np

# Part 1: Creating and Manipulating Tensors
print("PART 1: CREATING AND MANIPULATING TENSORS")
print("-----------------------------------------")

# 1.1 Creating tensors from Python lists
print("1.1 Creating tensors from different data sources:")
# Create a 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D tensor: {tensor_1d}")
print(f"Shape: {tensor_1d.shape}")

# Create a 2D tensor (matrix)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D tensor:\n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}")

# Create tensor from NumPy array
numpy_array = np.array([[1.5, 2.5], [3.5, 4.5]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy:\n{tensor_from_numpy}")
print(f"Data type: {tensor_from_numpy.dtype}")

# 1.2 Tensor creation functions
print("\n1.2 Tensor creation functions:")
# Create tensors with specific values
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.rand(2, 3)  # Uniform distribution [0, 1)

print(f"Zeros tensor:\n{zeros}")
print(f"Ones tensor:\n{ones}")
print(f"Random tensor:\n{random_tensor}")

# Create a range of values
range_tensor = torch.arange(0, 10, step=2)
print(f"Range tensor: {range_tensor}")

# Create tensor with specific data type
float_tensor = torch.ones(2, 2, dtype=torch.float32)
int_tensor = torch.ones(2, 2, dtype=torch.int32)
print(f"Float tensor:\n{float_tensor}")
print(f"Integer tensor:\n{int_tensor}")

# 1.3 Tensor reshaping
print("\n1.3 Tensor reshaping:")
original = torch.arange(12)
print(f"Original tensor: {original}, shape: {original.shape}")

# Reshape to 3x4 matrix
reshaped = original.reshape(3, 4)
print(f"Reshaped to 3x4:\n{reshaped}")

# Reshape using view (shares the same memory)
viewed = original.view(4, 3)
print(f"Viewed as 4x3:\n{viewed}")

# Transpose a tensor
transposed = reshaped.T
print(f"Transposed:\n{transposed}")

# Flatten a tensor
flattened = transposed.flatten()
print(f"Flattened: {flattened}")

# Part 2: Basic Tensor Operations
print("\nPART 2: BASIC TENSOR OPERATIONS")
print("------------------------------")

# 2.1 Element-wise operations
print("2.1 Element-wise operations:")
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")

# Addition
print(f"a + b:\n{a + b}")
print(f"torch.add(a, b):\n{torch.add(a, b)}")

# Subtraction
print(f"a - b:\n{a - b}")

# Multiplication (element-wise)
print(f"a * b:\n{a * b}")

# Division
print(f"a / b:\n{a / b}")

# Power
print(f"a^2:\n{a ** 2}")

# 2.2 Matrix operations
print("\n2.2 Matrix operations:")
# Matrix multiplication
mat_mul = torch.matmul(a, b)
print(f"Matrix multiplication (a @ b):\n{mat_mul}")
print(f"Same using torch.matmul(a, b):\n{torch.matmul(a, b)}")

# Dot product of 1D tensors
c = torch.tensor([1, 2, 3])
d = torch.tensor([4, 5, 6])
print(f"Dot product of {c} and {d}: {torch.dot(c, d)}")

# 2.3 Statistical operations
print("\n2.3 Statistical operations:")
sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(f"Sample tensor:\n{sample_tensor}")

print(f"Sum of all elements: {sample_tensor.sum()}")
print(f"Mean value: {sample_tensor.mean()}")
print(f"Standard deviation: {sample_tensor.std()}")
print(f"Maximum value: {sample_tensor.max()}")
print(f"Minimum value: {sample_tensor.min()}")

# Sum along specific dimensions
print(f"Sum along rows (dim=0):\n{sample_tensor.sum(dim=0)}")
print(f"Sum along columns (dim=1):\n{sample_tensor.sum(dim=1)}")

# Part 3: Tensor Operations for Machine Learning
print("\nPART 3: TENSOR OPERATIONS FOR MACHINE LEARNING")
print("--------------------------------------------")

# 3.1 Broadcasting
print("3.1 Broadcasting:")
matrix = torch.rand(3, 4)
vector = torch.rand(4)

print(f"Matrix shape: {matrix.shape}")
print(f"Vector shape: {vector.shape}")

# Broadcasting the vector to each row of the matrix
result = matrix + vector
print(f"Result shape after broadcasting: {result.shape}")
print(f"First few values:\n{result[:2, :2]}")

# 3.2 Indexing and slicing
print("\n3.2 Indexing and slicing:")
data = torch.arange(16).reshape(4, 4)
print(f"Original data:\n{data}")

# Select specific elements
print(f"Element at position [1, 2]: {data[1, 2]}")
print(f"First row: {data[0]}")
print(f"Last column: {data[:, -1]}")

# Slicing
print(f"Top-left 2x2 block:\n{data[:2, :2]}")
print(f"Bottom-right 2x2 block:\n{data[2:, 2:]}")

# Advanced indexing
indices = torch.tensor([0, 2])
print(f"Rows 0 and 2:\n{data[indices]}")

# 3.3 Device management (CPU/GPU)
print("\n3.3 Device management:")
# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# Create a tensor on the specified device
x = torch.rand(3, 3, device=device)
print(f"Tensor created on {device}:\n{x}")

# Move existing tensor to device
y = torch.rand(3, 3)
y_device = y.to(device)
print(f"Tensor moved to {device}")

# Part 4: Exercises
print("\nPART 4: EXERCISES")
print("---------------")
print("Complete the following exercises:")

# Exercise 1: Create a 3x3 identity matrix using PyTorch functions
print("\nExercise 1: Create a 3x3 identity matrix")
# TODO: Your code here
# identity_matrix = ...

# Exercise 2: Perform a matrix multiplication between a 2x3 and a 3x2 matrix
print("\nExercise 2: Matrix multiplication")
# TODO: Your code here
# matrix_a = ...
# matrix_b = ...
# result = ...

# Exercise 3: Calculate the cosine similarity between two vectors
print("\nExercise 3: Cosine similarity")
# TODO: Your code here
# vector_1 = ...
# vector_2 = ...
# cosine_similarity = ...

# Exercise 4: Normalize a matrix along rows (each row should sum to 1)
print("\nExercise 4: Normalize matrix rows")
# TODO: Your code here
# matrix = ...
# normalized = ...

# Exercise 5: Create a function that calculates the Euclidean distance between two points represented as tensors
print("\nExercise 5: Euclidean distance function")
# TODO: Your code here
# def euclidean_distance(point1, point2):
#     ...

# Part 5: Visualization of tensor operations
print("\nPART 5: VISUALIZATION")
print("-------------------")

# Create a simple 2D tensor for visualization
visual_tensor = torch.linspace(0, 10, 100).reshape(10, 10)

# Plot the tensor as a heatmap
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(visual_tensor, cmap='viridis')
plt.colorbar()
plt.title('Original Tensor')

# Apply some operations (e.g., Gaussian blur using convolution)
kernel = torch.ones(3, 3) / 9.0  # Simple 3x3 averaging kernel
kernel = kernel.view(1, 1, 3, 3)
visual_tensor_expanded = visual_tensor.view(1, 1, 10, 10)

# We need padding for convolution to maintain the size
padded = torch.nn.functional.pad(visual_tensor_expanded, (1, 1, 1, 1), 'reflect')
if torch.__version__ >= "1.8.0":  # For newer PyTorch versions
    blurred = torch.nn.functional.conv2d(padded, kernel)[0, 0]
else:
    # Manual convolution for older PyTorch
    blurred = torch.zeros(10, 10)
    for i in range(10):
        for j in range(10):
            blurred[i, j] = padded[0, 0, i:i+3, j:j+3].sum() / 9.0

plt.subplot(1, 2, 2)
plt.imshow(blurred.detach(), cmap='viridis')
plt.colorbar()
plt.title('Blurred Tensor (Convolution)')

plt.tight_layout()
plt.savefig('tensor_visualization.png')
plt.show()



print("\nPART 6: NEURAL NETWORK – ADD MORE LAYERS (Activity 1)")
print("------------------------------------------------------")

import torch.nn as nn
import torch.nn.functional as F

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


input_size = 784          
hidden1_size = 256
hidden2_size = 128
hidden3_size = 64
output_size = 10          


model = FeedforwardNN(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)

print(model)

dummy_input = torch.randn(1, input_size)
output = model(dummy_input)
print("Output shape:", output.shape)
