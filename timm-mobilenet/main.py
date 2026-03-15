import os
from src.config import get_model_config
from src.model import predict
from src.utils import load_image, get_imagenet_labels
from src.plotting import plot_results

def main():
    """
    Main function to run the image classification and plot the results.
    """
    image_path = os.path.join('images', 'f1.jpg')
    
    # Load image
    image = load_image(image_path)

    # Get model and data configuration
    model, data_config = get_model_config()

    # Make prediction
    top5_probabilities, top5_idx = predict(model, data_config, image)

    # Get ImageNet labels
    labels = get_imagenet_labels()

    # Plot results
    plot_results(image, top5_probabilities, top5_idx, labels)

    print('Top 5 probabilities:', top5_probabilities)
    print('Top 5 indices:', top5_idx)

if __name__ == "__main__":
    main()
