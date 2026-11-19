import torchvision
from lightly.transforms import SimCLRTransform

from src.fyp import utils

utils.seed_everything(42)

def main():


    transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
    dataset = torchvision.datasets.CIFAR10(
        "data/cifar10", download=True, transform=transform
    )



