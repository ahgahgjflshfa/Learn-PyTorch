# 02. Neural Network classification with PyTorch
# Classification is a problem of predicting whether something is on thing or another (there can be multiple things as the options).

import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
from helper_functions import accuracy_fn
import matplotlib.pyplot as plt

'''
1. Make classification data and get it ready
'''

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=42)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# Split data into training and test sets (split randomly)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

'''
2. Building a model

2.1 Setup a device agnostic code
2.2 Construct a model (by subclassing nn.Module)
2.3 Define a loss function and optimizer
2.4 Create a training loop and test loop
'''

# Setup a device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2.2 Create a model
# 2.2.1 Subclasses `nn.Module`
# 2.2.2 Create 2 `nn.Linear()` layers that are capable of handling the shapes of our data
# 2.2.3 Define a `forward()` method that outlines the forward pass (or forward computation) of the model
# 2.2.4 Instantiate the model and move it to the `device`

# 2.2.1 Subclasses `nn.Module`
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # 2.2.2 Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # hidden layer
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # output layer

    # 2.2.3 Define a forward method that outlines the forward pass (or forward computation) of the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2  -> output

# 2.2.4 Instantiate the model and move it to the `device`
model_0 = CircleModelV0().to(device)

# 2.3 Setup loss function and optimizer

# Setup loss function
# loss_fn = nn.BCELoss() # BCELoss requires inputs to have gone through sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss is built-in with sigmoid activation function

# Setup optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# 2.4 Create a training loop and test loop
# 1. Forward pass
# 2. Calculate the loss
# 3. Optimizer zero grad
# 4. Perform backpropagation
# 5. Optimizer step (gradient decent)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training loop
# for epoch in range(epochs):
#     ## Training
#     model_0.train()
#
#     # 1. Forward pass
#     y_logits = model_0(X_train).squeeze() # squeeze removes dimensions of size 1
#     y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels
#
#     # 2. Calculate the loss
#     loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss expects raw logits as input
#     # loss = loss_fn(torch.sigmoid(y_logits), y_train) # BCELoss expects prediction probability as input
#
#     # 3. Optimizer zero grad
#     optimizer.zero_grad()
#
#     # 4. Backpropagation
#     loss.backward()
#
#     # 5. Optimizer step
#     optimizer.step()
#
#     ## Testing
#     model_0.eval()
#     with torch.inference_mode():
#         # 1. Forward pass
#         test_logits = model_0(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
#
#     # 2. Calculate the loss
#     test_loss = loss_fn(test_logits, y_test)
#
#     # Print out what's happening
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Plot the decision boundary of the model
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plot_decision_boundary(model_0, X_train, y_train)
# plt.title("Training data")
# plt.subplot(1, 2, 2)
# plot_decision_boundary(model_0, X_test, y_test)
# plt.title("Test data")
# plt.show()

"""
Improving a model (from a model's perspective):
1. Add more layers - give the model more chances to learn about patterns in the data
2. Add more hidden units - go from 5 hidden units to 10 hidden units
3. Fit for longer - go from 100 epochs to 1000 epochs
4. Change/add activation functions
5. Change the learning rate
6. Change the optimizer
7. Change the loss function

These options are all from a model's perspective because they deal directly with the model, rather than the data.
And because these options are all values we (as machine learning engineers and data scientists) can change, they are 
called hyperparameters.
"""

# Adding more hidden units: 5 -> 10
# Increase the number of layers: 2 -> 3
# Increase the number of epochs: 100 -> 1000
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.layer_2(self.layer_1(x))) # this way of writing operations leverages speed-ups where possible behind the scenes

model_1 = CircleModelV1().to(device)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

# Create a training loop and test loop
# Train for longer
epochs = 1000

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # Training
    model_1.train()

    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate the loss
    loss = loss_fn(y_logits, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Test loop
    # model_1.eval()
    # with torch.inference_mode():
    #     test_logits = model_1(X_test).squeeze()
    #     test_pred = torch.round(torch.sigmoid(test_logits))
    #     test_loss = loss_fn(test_logits, y_test)
    #
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Plot the decision boundary of the model
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plot_decision_boundary(model_1, X_train, y_train)
# plt.title("Training data")
# plt.subplot(1, 2, 2)
# plot_decision_boundary(model_1, X_test, y_test)
# plt.title("Test data")
# plt.show()

"""
5.1 Preparing data to see if our model can fit a straight line
One way to troubleshoot to a larger problem is to test out a smaller problem.
"""

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

# Create train and test splits
train_split = int(0.8 * len(X_regression))
X_train_regression, X_test_regression = X_regression[:train_split], X_regression[train_split:]
y_train_regression, y_test_regression = y_regression[:train_split], y_regression[train_split:]

"""
5.2 Adjusting `model_1` to fit a straight line
"""

# Same architecture as model_1 (but using nn.Sequential())
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

# Set the number of epochs
epochs = 1000

# Put the data on the target device
X_train_regression, X_test_regression = X_train_regression.to(device), X_test_regression.to(device)
y_train_regression, y_test_regression = y_train_regression.to(device), y_test_regression.to(device)

# Training
# for epoch in range(epochs):
#     model_2.train()
#
#     # 1. Forward pass
#     y_pred = model_2(X_train_regression)
#
#     # 2. Calculate the loss
#     loss = loss_fn(y_pred, y_train_regression)
#
#     # 3. Optimizer zero grad
#     optimizer.zero_grad()
#
#     # 4. Loss backward (backpropagation)
#     loss.backward()
#
#     # 5. Optimizer step
#     optimizer.step()
#
#     # Test loop
#     model_2.eval()
#     with torch.inference_mode():
#         test_pred = model_2(X_test_regression)
#         test_loss = loss_fn(test_pred, y_test_regression)
#
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

'''
6. The missing piece: non-linearity

"What patterns could you draw if you were given an infinite amount of a straight and non-straight lines?"

or in machine learning terms, an infinite (but really it is finite) of linear and non-linear functions?
'''

"""
6.1 Recreating non-linear data (circles)
"""

# Make and plot data
n_samples = 1000

X, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=42)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Convert data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
6.2 Building a model with non-linearity

* Linear = Straight line
* Non-linear = Curved line

Artificial neural networks are a large combination of linear (straight) and non-straight (curved) functions which are
potentially able to find patterns in data.
"""

# Build a model with non-linear activation functions
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()   # ReLU is a non-linear activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),
                            lr=0.4)

"""
6.3 Training model with non-linearity
"""

# Put all data on target device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

epochs = 1000

for epoch in range(epochs):
    # Training
    model_3.train()

    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate the loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Test
    model_3.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate the loss and acc
        test_loss = loss_fn(test_pred, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f} | Test loss: {test_loss:.4f} | Test Acc: {test_acc}")

"""
6.4 Evaluating model trained with non-linear activation functions
"""

# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# Plot decision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Model_1 (no non-linearity)")
plot_decision_boundary(model_1, X_test, y_test)
plt.subplot(1, 2, 2)
plt.title("Model_3")
plot_decision_boundary(model_3, X_test, y_test)
plt.show()