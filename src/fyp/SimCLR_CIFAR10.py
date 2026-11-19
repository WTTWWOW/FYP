import torchvision
from lightly.transforms import SimCLRTransform, utils

from fyp import utils_src


def main():
    if __name__ == "__main__":
        utils_src.seed_everything(42)
        transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
        
        dataset = torchvision.datasets.CIFAR10(
            "data/cifar10", download=True, transform=transform
        )



