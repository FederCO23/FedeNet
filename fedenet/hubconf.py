from fedenet import build_model
from fedenet.weights import load_rgb_pretrained_into_fedenet


def fedenet_tiny(pretrained: bool = True, map_location="cpu"):
    model = build_model()
    model = model.to(map_location)
    if pretrained:
        load_rgb_pretrained_into_fedenet(model, device=map_location)
    model.eval()
    return model
