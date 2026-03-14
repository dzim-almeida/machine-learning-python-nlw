# LeNet MNIST Classifier

This project is a Python implementation of the LeNet-5 convolutional neural network for classifying handwritten digits from the MNIST dataset. It's built using PyTorch.

## Project Structure

```
lenet/
├── .venv/                # Virtual environment
├── data/                 # MNIST dataset will be downloaded here
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Handles data loading and preprocessing
│   ├── model.py          # Defines the LeNet architecture
│   ├── predict.py        # Logic for making predictions on new images
│   └── train.py          # Training and evaluation logic
├── main.py               # Main script to train or predict
├── requirements.txt      # Project dependencies
└── README.md
```

## Setup

1.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the following command from the `lenet` directory:

```bash
python main.py train
```

This will download the MNIST dataset, train the LeNet model for 10 epochs, and save the trained model weights to a file named `lenet_mnist.pth`.

### Prediction

To make a prediction on a new image of a digit, use the `predict` command. You'll need to provide the path to your image.

```bash
python main.py predict --image /path/to/your/digit_image.png
```

The script will display the image along with the model's prediction.

## How it Works

1.  **`dataset.py`**: Downloads the MNIST dataset using `torchvision.datasets.MNIST`. It also defines the necessary transformations to convert the images into tensors and normalize them.
2.  **`model.py`**: Implements the LeNet-5 architecture using `torch.nn.Module`. It consists of two convolutional layers followed by three fully connected layers.
3.  **`train.py`**:
    *   Initializes the `LeNet` model, the `CrossEntropyLoss` function, and the `Adam` optimizer.
    *   Loops through the training data for a specified number of epochs.
    *   For each batch, it performs a forward pass, calculates the loss, performs a backward pass (backpropagation), and updates the model's weights.
    *   After training, it evaluates the model's accuracy on the test set.
    *   Saves the trained model's state dictionary.
4.  **`predict.py`**:
    *   Loads the saved model weights.
    *   Opens and preprocesses the input image to match the format expected by the model.
    *   Performs a forward pass to get the prediction.
    *   Displays the result.
5.  **`main.py`**: Uses Python's `argparse` module to create a simple command-line interface that calls the appropriate functions from `train.py` or `predict.py`.
