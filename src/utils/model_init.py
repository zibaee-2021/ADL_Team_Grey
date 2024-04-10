import torch.nn as nn

def initialise_weights(model):
    """
    Random initialisation of model weights
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name or 'fc' in name:
                nn.init.xavier_uniform_(param)
            elif 'conv' in name:  # Initialize convolutional layers
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)  # Default initialization for other layers
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)  # Initialize biases to zero

