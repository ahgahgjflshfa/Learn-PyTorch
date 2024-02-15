import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from helper_functions import accuracy_fn, print_train_time, eval_model
import matplotlib.pyplot as plt
from timeit import default_timer
from tqdm.auto import tqdm
from pathlib import Path
import os

"""
0. Computer vision libraries in PyTorch

* `torchvision` - Base domain library for PyTorch computer vision
* `torchvision.datasets` - Get datasets and data loading functions for computer vision here
* `torchvision.models` - Get pre-trained computer vision models that you can leverage for your own problems
* `torchvision.transforms` - Functions for manipulating your vision data (images) to be suitable for use with an ML model
* `torchvision.utils.data.Dataset` - Base dataset class for PyTorch
* `torchvision.utils.data.DataLoader` - Creates a Python iterable over a dataset
"""

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
1. Getting datasets

The dataset we'll be using is FashionMNIST from torchvision.datasets
"""

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",    # where to save the dataset?
    train=True, # do we want the training dataset?
    download=True,  # do we want to download yes/no?
    transform=ToTensor(),   # how do we want to transform the data?
    target_transform=None   # how do we want to transform the labels/targets?
)

# Setup test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

class_names = train_data.classes

# 1.2 Visualizing the data
# fig = plt.figure(figsize=(10, 10)) # `figsize` decides the size of the figure
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis("off")
#
# plt.show()

"""
2. Prepare DataLoader

Right now, our data is in the form of PyTorch Datasets.

We want to change Datasets into DataLoader.

DataLoader turns our dataset into a Python iterable.

More specifically, we want to turn our data into batches (or mini-batches).

Why would we do this?
    1. It is more computationally efficient, as in, your computing hardware may not be able to look (store in memory) at
    60000 images in one hit. So we break it down to 32 images at a time (batch size of 32).

    2. It gives our neural network more chances to update its gradients per epoch.
"""

# Setups the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

"""
3. Model 0: Build a baseline model

When starting to build a series of machine learning model experiments, it's best practice to start with a baseline
model.

A baseline model is a simple model you will try and improve upon with subsequent models/experiments.

In other words: start simply and add complexity when necessary.
"""

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units: int = 8):
        """Initialize multi-class classification model

        Args:
            input_shape (int): Number of input features to the model.
            hidden_units (int): Number of hidden units between layers.
            output_shape (int): Number of output features of the model (how many classes there are).

        Returns:
            None

        Example:
            None
        """
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)

torch.cuda.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=784,  # this is 28*28
                              output_shape=len(class_names),
                              hidden_units=8)

"""
3.1 Setup loss, optimizer and evaluation metrics

* Loss function - Since we're working with multi-class data, our loss function will be `nn.CrossEntropy()`.
* Optimizer - Our optimizer `torch.optim.SGD()`.
* Evaluation metric - Sice we're working on a classification problem, let's use accuracy as our evaluation metric.
"""

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

"""
3.2 Creating a function to time our experiments

Machine learning is very experimental.

Two of the main things you'll often want to track are:
1. Model's performance (loss and accuracy values etc)
2. How fast it runs
"""

"""
3.3 Creating a training loop and training a model on batches of data

1. Loop through epochs.
2. Loop through training batches, perform training steps, calculate the train loss *per batch*.
3. Loop through testing batches, perform testing steps, calculate the test loss *per batch*.
4. Print out what's happening.
5. Time it all (for fun).
"""

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start = default_timer()

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 0

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")

    # Train
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        # 1. Forward pass
        y_logits = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_logits, y)
        train_loss += loss  # accumulate train loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out what's happening
        # if batch % 400 == 0:
        #     print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    # Test
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            test_logits = model_0(X_test)

            # 2. Calculate loss (cumulatively)
            test_loss += loss_fn(test_logits, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_logits.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

    # print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

# Calculate training time
train_time_end = default_timer()
total_traintime_model_0 = print_train_time(start=train_time_start,
                                           end=train_time_end,
                                           device='cpu')

# 4. Make prediction and get Model 0 results
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             acc_fn=accuracy_fn)

"""
6. Model 1: Building a better model with non-linearity
"""

os.system('cls')

# Create a model with linear and non-linear layer
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)

# Create an instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, hidden_units=8, output_shape=len(class_names)).to(device)

# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

"""
6.2 Functionizing training and evaluation/testing loop

Let's create function for :
* training loop - `train_step()`
* testing loop - `test_step()`
"""

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn,
               device: str = 'cpu'):
    """Performs a training with model trying to learn on `data_loader`

    Args:
        model: The model to train.
        data_loader: The dataloader which stores the training data.
        loss_fn: The loss function to evaluate the loss.
        optimizer: The optimizer used to optimizer the model.
        acc_fn: The accuracy function to evaluate the accuracy of a model.
        device: The device to use.

    Returns:
        None
    """
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)



        # 1. Forward pass (outputs the raw logits from the model)
        y_logits = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_logits, y)
        train_loss += loss  # accumulate train loss
        train_acc += acc_fn(y_true=y,
                                 y_pred=y_logits.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"\nTrain loss: {train_loss:.5f} | Train acc: {train_acc:.2f}", end="")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn,
              device: str = 'cpu'):
    """Performs a testing loop step on model going over `data_loader`

    Args:
        model: The model to test.
        data_loader: The dataloader which stores the test data.
        loss_fn: The loss function to evaluate the loss.
        acc_fn: The accuracy function to evaluate the accuracy of a model.
        device: The device to use.

    Returns:
        None
    """
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            # Put data in `device`
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            test_logits = model(X)

            # 2. Calculate loss (cumulatively)
            test_loss += loss_fn(test_logits, y)
            test_acc += acc_fn(y_true=y,
                                    y_pred=test_logits.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(data_loader)

        # Calculate the test acc average per batch
        test_acc /= len(data_loader)

    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}")

train_time_start_on_gpu = default_timer()

epochs = 0

for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}--------")
    # Train
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               acc_fn=accuracy_fn,
               device=device)
    # Test
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              acc_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = default_timer()

train_time_on_gpu = print_train_time(start=train_time_start_on_gpu,
                                     end=train_time_end_on_gpu,
                                     device=device)

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             acc_fn=accuracy_fn,
                             device=device)

os.system('cls')

"""
Model 2: Building a Convolutional Neural Network (CNN)

CNN's are also know as ConvNets.
CNN's are know for their capabilities to find patterns in visual data.
"""

# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture that replicates the TinyVGG model from CNN explainer website
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,   # there's a trick calculating this
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        # print(f"Output shape of X: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of X: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of X: {x.shape}")
        return x

# torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(),
                            lr=0.1)

epochs = 3

start_time = default_timer()

for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n--------")
    # Train
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               acc_fn=accuracy_fn,
               device=device)

    # Test
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              acc_fn=accuracy_fn,
              device=device)

end_time = default_timer()
print_train_time(start_time, end_time, device=device)

model_2_results = eval_model(model=model_2,
                              data_loader=test_dataloader,
                              loss_fn=loss_fn,
                              acc_fn=accuracy_fn,
                              device=device)

os.system("cls")

"""
10. Making Confusion matrix for further prediction evaluation

A confusion matrix is a fantastic way of evaluating your classification

1. Make predictions with our trained model on the test dataset
2. Create a confusion matrix `torchmetrics.ConfusionMatrix`
3. Plot the confusion matrix using `mlxtend.plotting.plot_confusion_matrix`
"""

# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs raw logits)
        y_logits = model_2(X)

        # Turn logits into predictions
        y_pred = y_logits.argmax(dim=1)

        y_preds.append(y_pred.cpu())

# Concatenate all predictions into a single tensor
y_preds_tensor = torch.cat(y_preds)

# Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds_tensor,
                         target=test_data.targets)


# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)

plt.show()

"""
11. Save and load best performing model
"""

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

# Create a new instance of FashionMNISTModelV2
model_2_load = FashionMNISTModelV2(input_shape=1,
                                   hidden_units=10,
                                   output_shape=len(class_names))

# Load in the saved state_dict()
model_2_load.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send the model to the target device
model_2_load = model_2_load.to(device)

# Evaluate the loaded model
torch.manual_seed(42)
model_2_load_results = eval_model(model=model_2_load,
                                  data_loader=test_dataloader,
                                  loss_fn=loss_fn,
                                  acc_fn=accuracy_fn,
                                  device=device)

print(f"Model 2 results: {model_2_results}")
print(f"Model 2 load results: {model_2_load_results}")