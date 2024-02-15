import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Plot function
def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={'size': 14})

    # Show the plot
    plt.show()

# Function to create and return a torch.nn.Module model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Create some data using the linear regression formula of y = weight * x + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# Split data into training and testing
train_split = 0.8
train_split_index = int(train_split * len(X))
X_train, y_train = X[:train_split_index], y[:train_split_index]
X_test, y_test = X[train_split_index:], y[train_split_index:]

# Plot the data
# plot_predictions(X_train, y_train, X_test, y_test)

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2().to(device)

### Training
"""
For training we need:
* Loss function (how far off is the prediction from the actual label)
* Optimizer (how to update the model parameters)
* Training loop
* Testing loop
"""

# Setup the loss function
loss_fn = nn.L1Loss()

# Setup the optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

# Put data on device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training loop
torch.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: | loss: {loss} | test_loss: {test_loss}")
        print(X_train)

# Plot the predictions
model_1.eval()
with torch.inference_mode():
    test_pred = model_1(X_test)

plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), test_pred.cpu())

# Save the model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Load a model
loaded_model_1 = LinearRegressionModelV2().to(device)

loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(f"Model loaded from {MODEL_SAVE_PATH}")