import torch
from src.model import LeNet
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def predict(image_path, model_path='lenet_mnist.pth'):
    """
    Loads a trained model and makes a prediction on a single image.
    """
    # Load the model
    model = LeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the image
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        
    # Display the image and prediction
    plt.imshow(Image.open(image_path), cmap='gray')
    plt.title(f'Prediction: {predicted.item()}')
    plt.show()

    return predicted.item()
