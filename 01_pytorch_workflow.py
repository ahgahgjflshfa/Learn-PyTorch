import torch
import matplotlib.pyplot as plt
from pathlib import Path

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(1,
                                                      requires_grad=True,
                                                      dtype=torch.float))

        self.bias = torch.nn.Parameter(torch.randn(1,
                                                    requires_grad=True,
                                                    dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights * x + self.bias # this is the linear regression formula

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

torch.manual_seed(42)
model_0 = LinearRegressionModel()

weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02, dtype=torch.float32).unsqueeze(1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Setup a loss function
loss_fn = torch.nn.L1Loss()

# Setup an optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

# Track different values
epoch_values = []
loss_values = []
test_loss_values = []

# Building a training loop and a testing loop
epochs = 200

### Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # train mode in PyTorch sets all parameters that require gradients to requires_grad=True

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # By default, how the optimizer changes will accumulate through the loop, so... we have to zero them above in step 3 for the next iteration of the loop

    ###Testing
    model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
    with torch.inference_mode(): # turns off gradient tracking & a couple more things behind the scenes
        # 1. Do the forward pass
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        # Track the values
        epoch_values.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())

        # Print out the loss every 10 epochs
        print(f"Epoch {epoch}: | loss: {loss} | test_loss: {test_loss}")

        # Print out model state_dict()
        print(model_0.state_dict())

        print()

# Plot the loss curves
plt.plot(epoch_values, loss_values, label="Training loss")
plt.plot(epoch_values, test_loss_values, label="Testing loss")
plt.title("training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

with torch.inference_mode(): # turns off gradient tracking & a couple more things behind the scenes
    y_pred = model_0(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_pred)

### Save the model
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state_dict()
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

### Load the model
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the model state_dict() (this will update the new instance of our model with the weights from the saved model)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Test the loaded model
model_0.eval()
loaded_model_0.eval()
with torch.inference_mode(): # turns off gradient tracking & a couple more things behind the scenes
    y_pred = model_0(X_test)
    y_pred_loaded = loaded_model_0(X_test)

print(f"y_pred: {y_pred}")
print(f"y_pred_loaded: {y_pred_loaded}")
print(y_pred == y_pred_loaded)