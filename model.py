import torch.nn as nn
from torchvision import models

def get_efficientnet_v2(num_classes=5, pretrained=True):
    """
    Loads a pretrained EfficientNetV2-Small model and replaces the final
    classifier layer to match the number of classes.
    """
    #  Load the pretrained EfficientNetV2 model
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    
    # 2. The final layer in EfficientNet is different from ResNet.
    # It's inside the 'classifier' attribute. We need to find the
    # number of input features to the last Linear layer.
    num_ftrs = model.classifier[-1].in_features
    
    # 3. Replace the final layer with a new one for our 5 classes
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    return model
