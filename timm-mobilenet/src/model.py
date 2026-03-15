import torch
import timm

def predict(model, data_config, image):
    """Generates predictions for a given image.

    Args:
        model: The pre-trained model.
        data_config (dict): The data configuration for the model.
        image (PIL.Image.Image): The input image.

    Returns:
        tuple: A tuple containing the top 5 probabilities and indices.
    """
    transform = timm.data.create_transform(**data_config, is_training=False)
    output = model(transform(image).unsqueeze(0))
    top5_probabilities, top5_idx = torch.topk(output.softmax(dim=1) * 100, k=5)
    return top5_probabilities, top5_idx