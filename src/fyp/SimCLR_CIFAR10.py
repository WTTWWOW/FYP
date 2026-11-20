import csv
import json
from datetime import datetime
from pathlib import Path

import torch.utils.data
import torchvision
from lightly.transforms import utils
from lightly.transforms.simclr_transform import SimCLRTransform
from tqdm import tqdm

from fyp import config, modules, utils_src


def main():
        
    utils_src.seed_everything(42)

    run_dir = Path("outputs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    cfg = {
        **{k: v for k, v in vars(config).items() if not k.startswith("_")},
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    raw_data = torchvision.datasets.CIFAR10("data/cifar10", download=True)
    
    transform = SimCLRTransform(input_size=config.input_size, vf_prob=config.vf_prob, rr_prob=config.rr_prob)
    ##Transform to normalise testing set
    test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((config.input_size, config.input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
    )

    dataset_train = modules.transformed_dataset(raw_data, transform)
    dataset_test = modules.transformed_dataset(raw_data, test_transform)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules.simclr_model().to(device)
    loss_fn = model.loss_fn
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{config.max_epochs}", leave=False)
        
        for (x0, x1),  _ in progress:
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0), model(x1)
            loss = loss_fn(z0, z1)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        average_loss = total_loss / len(dataloader_train)
        with open(run_dir / "metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, average_loss])
        tqdm.write(f"Epoch {epoch + 1}/{config.max_epochs}, Loss: {average_loss:.4f}")

    torch.save(model.backbone.state_dict(), run_dir / "model.pt")


    
            
    



if __name__ == "__main__":
    main()


