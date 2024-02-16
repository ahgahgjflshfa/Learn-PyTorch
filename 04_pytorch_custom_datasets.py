import os
import torch
import random
import zipfile
import requests
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.io
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchinfo import summary
from helper_functions import train
from timeit import default_timer as timer
from torchvision import datasets, transforms, io
from torch.utils.data import Dataset, DataLoader
from helper_functions import plot_transformed_images, accuracy_fn, print_train_time
from helper_functions import pred_and_plot_image

"""
Chapter 4. PyTorch Custom Datasets

We've used some datasets with PyTorch before.
But how do you get your own data into PyTorch?
One of the ways to da so is via: custom datasets.

## Domain library

Depending on what you're working on, vision, text, audio, recommendation, you'll want to look into each of the PyTorch
domain libraries for existing data loading functions and customizable data loading functions.
"""

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

"""
1. Get data

Our dataset is a subset of the Food101 dataset.
Food101 starts 101 different classes of food and 1000 images per class (750 training, 250 testing).
Our dataset starts with 3 classes of food and only 10% of the images (~75 training, 25 testing).

Why do this?
When starting out ML projects, it's important to try things on a small scale and then increase the scale when necessary.
The whole point is to speed up how fast you can experiment.
"""

# Setup path to a data folder
data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"

# # If the image folder doesn't exist, download it and prepare it...
# if image_path.is_dir():
#     print(f"{image_path} directory exists...skipping download")
# else:
#     print(f"Did not find {image_path} directory, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)
#
# # Download pizza, steak, sushi data
# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#     print("Downloading pizza, steak, sushi data...")
#     f.write(request.content)
#
# # Unzip pizza, steak, sushi data
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#     print("Unzipping pizza, steak, sushi data...")
#     zip_ref.extractall(image_path)

"""
2. Becoming one with the data (data preparation and data exploration)
"""

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

"""
2.1 Visualizing an image

Let's write some code to:
    1. Get all of the image paths
    2. Pick a random image path using Python's random.choice()
    3. Get the image class name using pathlib.Path.parent.stem
    4. Since we're working with images, let's open the image with Python's PIL
    5. We'll then show the image and print metadata
"""

# Set seed
# random.seed(42)

# 1. Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Pick a random image path
random_image_path = random.choice(image_path_list)
# print(f"Random image path: {random_image_path}")

# 3. Getimage class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
# print(f"Image class: {image_class}")

# 4. Open image
img = Image.open(random_image_path)

# # 5. Print metadata
# print(f"Image Image Height: {img.height}")
# print(f"Image Image Width: {img.width}\n")

# Try to visualize image using matplotlib
# plt.imshow(img)
# plt.axis(False)
# plt.show()

"""
3. Transforming data

Before we can use our image data with PyTorch:
    1. Turn your target data into tensors (in our case, numerical representation of our images).
    2. Turn it into a `torch.utils.data.Dataset` and subsequently a `torch.utils.data.DataLoader`, we'll call these
    `dataset` and `dataloader`.
"""

# 3.1 Transforming data with `torchvision.transforms`

# Write a transform for image
data_transform = transforms.Compose([
    # Resize image to 64x64
    transforms.Resize((64, 64)),
    # Flip the images randomly on the horizontal axis
    transforms.RandomHorizontalFlip(p=0.5),
    # Convert image to tensor
    transforms.ToTensor()
])

# plot_transformed_images(image_paths=image_path_list, transform=data_transform, n=3, seed=42)

"""
4. Option 1: Loading image data using `ImageFolder`

We can load image classification data using `torchvision.datasets.ImageFolder`.
"""

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # a transform for the data
                                  target_transform=None)    # a transform for the label/target

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.classes

# Set batch size
BATCH_SIZE = 32

# 4.1 Turn datasets into dataloaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              # num_workers=1,  # Runtime error...
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             # num_workers=1,
                             shuffle=False)

# img, label = next(iter(train_dataloader))

# print(f"Image shape: {img.shape}")

"""
5. Option 2: Loading Image Data with a Custom `Dataset`

1. Want to be able to load images(or other types of data) from file
2. Want to be able to get class names from the Dataset
3. Want to be able to get classes as dictionary from the Dataset

Pros:
    * Can create a `Dataset` out of almost anything
    * Not limited to PyTorch pre-built `Dataset` functions
    
Cons:
    * Even though you could create `Dataset` out of almost anything, it doesn't mean it will work...
    * Using a custom `Dataset` often results in us writing more code, which could be prone to errors or performance
      issues.
"""

"""
5.1 Creating a helper function to get class names from root folder

We want a function to:

    1. Get the class names using `os.scandir()` to traverse a target directory (ideally the directory is in standard
       classification format).
    2. Raise an error if the class names aren't found (if this happens, there might be something wrong with the
       directory structure).
    3. Turn the class names into a dictionary and returns it.
"""

def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
    """Finds the class folder names in a target directory.

    Args:
        directory: The root directory to scan for class names.

    Returns:
        A tuple of (class_names, class_to_idx).
    """
    classes = sorted(d.name for d in os.scandir(directory) if d.is_dir())

    # Raise an error if no class names are found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

"""
5.2 Create a custom `Dataset` to replace `ImageFolder`

TO Create our own custom dataset, we want to:
1. Subclass `torch.utils.data.Dataset`
2. Init our subclass with a target directory (the directory we'd like to get data from) as well as a transform if we'd
   like to transform the data.
3. Create several attributes:
   * paths - paths of our images
   * transform - the transform we'd like to use
   * classes - a list of the target classes
   * class_to_idx - a dictionary mapping class names to indices(labels/targets)
4. Create a function to `load_images()`, this function will open an image.
5. Overwrite the `__len__()` method to return the length of our dataset. (Optional)
6. Overwrite the `__getitem__()` method to return a given sample when passed an index.
"""

# 1. Subclass `torch.utils.data.Dataset`
class ImageFolderCustom(torch.utils.data.Dataset):
    # 2. Initialize our custom dataset
    def __init__(self, root: str | Path, transform=None):
        # 3. Create class attributes
        # Get all the image paths
        self.paths = list(Path(root).glob("*/*.jpg"))

        # Setup transform
        self.transform = transform

        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(root)

    # 4. Create a function to load images
    def load_image(self, index: int) -> Image.Image:
        """Loads an image from a given index.

        Args:
            index: The index of the image to load.

        Returns:
            The image at the given index.
        """
        path = self.paths[index]
        image = Image.open(path)
        return image

    # 5. Overwrite the `__len__()` method to return the length of our dataset
    def __len__(self) -> int:
        return len(self.paths)

    # 6. Overwrite the `__getitem__()` method to return a given sample when passed an index
    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int]:
        """Returns one sample of data, data and label (X, y)"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # Expects path in format: data_folder(root)/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

# Create a transform
train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Test out ImageFolderCustom
train_data_custom = ImageFolderCustom(root=train_dir, transform=train_transform)

test_data_custom = ImageFolderCustom(root=test_dir, transform=test_transform)

"""
5.3 Create a function to display random images (in helper_function.py)

1. Take in a `Dataset` and a number of other parameters such as class names and how many images to visualize.
2. To prevent the display getting out of hand, let's cap the number of images to see at 10.
3. Set the random seed for reproducibility.
4. Get a list of random sample indices from the target dataset.
5. Setup a matplotlib plot
6. Loop through the random sample indices and plot them with matplotlib.
7. Make sure the dimensions of our images line up with matplotlib (HWC).
"""


# 5.4 Turn custom loaded images into `DataLoader`'s
BATCH_SIZE = 32
NUM_WORKERS = 1 # will cause runtime error when > 0 then use iter()

train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

"""
6. Other form of transform (data augmentation)

Data augmentation is the process of artificially adding diversity to your training data.

In the case of image data, this may mean applying various image transformations to the training images.

This practice hopefully results in a model that's more generalizable to unseen data.

Let's take a look at one particular type of data augmentation used to train PyTorch vision models to state of the art
levels...
"""

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

"""
7. Model 0: TinyVGG without data augmentation
"""

# 7.1 Create transforms and loading data for Model 0
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    # transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# 1. Load and transform data
train_data_simple =datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform,
    target_transform=None
)

test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform,
    target_transform=None
)

# 2. Turn the datasets into DataLoaders
BATCH_SIZE = 32
NUM_WORKERS = 0

# Create data loaders
train_data_loader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_data_loader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# 7.2 Create TinyVGG model class
class TinyVGG(nn.Module):
    """Model architecture copying TinyVGG from CNN Explainer"""
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units: int = 8) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # Default stride value is same as kernel size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 169,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))   # benefits from operator fusion

class_names = train_data_simple.classes

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,    # Number of colour channels
                  output_shape=len(class_names),
                  hidden_units=32).to(device)

# 7.4 Use `torchinfo` to get an idea of the shapes going through our model
# summary(model_0, [1, 3, 64, 64])

# Setup optimizer and loss function and epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=0.001)

# Start the timer
start_time = timer()

model_0_results = train(model=model_0,
                        epochs=0,
                        train_dataloader=train_data_loader_simple,
                        test_dataloader=test_data_loader_simple,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        acc_fn=accuracy_fn,
                        device=device)

end_time = timer()

# print_train_time(start_time, end_time, device)

"""
7.8 Plot the loss curves of Model 0

A **loss curve** is a way of tracking your model's progress over time.
"""

def plot_loss_curves(results: dict[str, list[float]]):
    """

    Args:
        results:

    Returns:

    """

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_acc = results["test_acc"]

    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

# plot_loss_curves(results)

"""
8. What should an ideal loss curve look like?

* https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic?hl=zh-tw

A loss curve is one of the most helpful ways to troubleshoot a model.
"""

"""
9. Model 1: TinyVGG with Data Augmentation

Now let's try another modeling experiment this time using the same model as before with some data augmentation.
"""

# 9.1 Create transform with data augmentation
BATCH_SIZE = 32
NUM_WORKERS = 0

# Create training transform with TrivialAugment
train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# 9.2 Create train and test `Dataset`'s and `DataLoader`'s with data augmentation
# Turn images folders into Datasets
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform_trivial,
                                  target_transform=None)

test_data_simple = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform_simple,
                                 target_transform=None)

# Turn datasets into data loader
train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3,
                  output_shape=len(class_names),
                  hidden_units=10).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.001)
NUM_EPOCHS = 0

model_1_results = train(model=model_1,
                        epochs=NUM_EPOCHS,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        acc_fn=accuracy_fn,
                        device=device)

# plot_loss_curves(results)

"""
10. Compare model results

After evaluating our modelling experiments on their own, it's important to compare them to each other.

There's a few different way to do this:
    1. Hard coding (what we're doing)
    2. PyTorch + Tensorboard - https://pytorch.org/docs/stable/tensorboard.html
    3. Weights & Biases - https://wandb.ai/site/experiment-tracking
    4. MLFlow - https://mlflow.org
"""

model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

# print(model_0_df)

# # Setup a plot
# plt.figure(figsize=(15, 6))
#
# epochs = range(len(model_0_df))
#
# plt.subplot(2, 2, 1)
# plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
# plt.title("Train Loss")
# plt.xlabel("Epochs")
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
# plt.title("Test Loss")
# plt.xlabel("Epochs")
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
# plt.title("Train acc")
# plt.xlabel("Epochs")
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
# plt.title("Test acc")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

"""
11. Making a prediction on a custom image

Although we've trained a model on custom data... how do you make a prediction on a sample/image that's not in either
training or testing dataset.
"""

# Download custom image
custom_image_path = data_path / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/blob/main/images/04-pizza-dad.jpeg?raw=true")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download...")

"""
11.1 Loading in a custom image with PyTorch

We have to make sure our custom image is in the same format as the data our model was trained on.
    * In tensor form with datatype (torch.float32)
    * Of shape (3, 64, 64)
    * On the right device

We can read an image into PyTorch using the `torchvision.io.read_image` function.
"""

custom_image_unit8 = io.read_image(str(custom_image_path))

# 11.2 Making a prediction on a custom image with a trained PyTorch model

# Try to make a prediction on an image in unit8 format
model_1.eval()
with torch.inference_mode():
    # model_1(custom_image_unit8.to(device))
    pass

custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 225. # Divide by 255 to get values between 0 and 1

with torch.inference_mode():
    # model_1(custom_image_unit8.to(device))
    pass

# Create transform pipeline to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64), antialias=False)
])

# Transformed target image
custom_image_transformed = custom_image_transform(custom_image)

model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

"""
Note, to make a prediction on a custom image we had to:

* Load the image and turn it into a tensor
* Make sure the image was the same datatype as the data our model (torch.float32).
* Make sure the image was the same shape as the data the model was trained on (3, 64, 64) with a batch size... (1, 3, 64, 64)
* Make sure the image was on the same device as our model.
"""

# print(class_names[custom_image_pred.argmax(dim=1).cpu()])   # put it into the cpu in case of errors

"""
11.3 Putting custom image prediction together: building a function

A function where we pass an image path to and have our model predict on that image and plot the image + prediction.
"""

pred_and_plot_image(model=model_1,
                    image_path=str(custom_image_path),
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)