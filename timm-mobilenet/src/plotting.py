import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Conteúdo a ser adicionado depois

def plot_results(image, probabilities, indices, labels):
    """Plots the input image and a bar chart with the top 5 predictions.

    Args:
        image (PIL.Image.Image): The input image.
        probabilities (torch.Tensor): The top 5 probabilities.
        indices (torch.Tensor): The top 5 indices.
        labels (list): A list of all class labels.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')

    # Plot the bar chart
    y_pos = np.arange(len(indices[0]))
    ax2.barh(y_pos, probabilities[0].detach().numpy())
    ax2.set_yticks(y_pos)
    class_labels = [labels[i] for i in indices[0]]
    ax2.set_yticklabels(class_labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top 5 Predictions')

    plt.tight_layout()
    plt.show()