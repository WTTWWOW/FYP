import torch
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch.utils.data import Dataset


#class for applying transforms to datasets
class transformed_dataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)

#class for simclr architecture
class simclr_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.loss_fn = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    