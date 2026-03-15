import timm


def get_model_config(model_name='mobilenetv3_small_100.lamb_in1k'):
    """Gets the model and its data configuration.

    Args:
        model_name (str, optional): The name of the model to use. 
            Defaults to 'mobilenetv3_small_100.lamb_in1k'.

    Returns:
        tuple: A tuple containing the model and data configuration.
    """
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    return model, data_config