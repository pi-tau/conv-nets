import csv
import json
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as tt


from googlenet import GoogLeNet_CIFAR10
from resnet import ResNet_CIFAR10
from densenet import DenseNet_CIFAR10


class Trainee(pl.LightningModule):

    def __init__(self, model, config):
        """Init a Lightning module."""
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.config["lr"], weight_decay=self.config["reg"])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y, reduction="mean")

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        acc = (pred.argmax(dim=-1) == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x).argmax(dim=-1)
        acc = (pred == y).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x).argmax(dim=-1)
        acc = (pred == y).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)


def load_CIFAR10():
    """Create train, validation and test dataloaders for the CIFAR10 dataset."""
    # First download the CIFAR10 training dataset and compute the means and
    # variances separately for each channel dimension.
    train_set = torchvision.datasets.CIFAR10("datasets", train=True, download=True)
    mean = (train_set.data / 255.0).mean(axis=(0, 1, 2))
    std = (train_set.data / 255.0).std(axis=(0, 1, 2))

    # The training, validation and test sets will be normalized with the
    # calculated statistics. In addition the training set will be augmented with
    # random flips and random crops. We will flip each image horizontally with a
    # 50% probability. Then we will randomly crop the image with the given scale
    # and aspect ratio, and rescale the crop afterwards to the original size.
    _, H, W, C = train_set.data.shape
    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        tt.ToTensor(),
        tt.Normalize(mean, std),
    ])

    # Create train, validation and test loaders with the defined transformations.
    # Note that the validation set is a split from the training set but is
    # transformed with the test transform, i.e. without random flips and crops.
    # Use a generator with a manual seed in order to crate identical splits.
    train_set = torchvision.datasets.CIFAR10("datasets", train=True, transform=train_transform)
    train_set, _ = data.random_split(train_set, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
    val_set = torchvision.datasets.CIFAR10("datasets", train=True, transform=test_transform)
    _, val_set = data.random_split(val_set, [0.8, 0.2], generator=torch.Generator().manual_seed(10))
    test_set = torchvision.datasets.CIFAR10("datasets", train=False, download=True, transform=test_transform)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader)


def train(model, args):
    train_loader, val_loader, test_loader = load_CIFAR10()

    trainee = Trainee(model, config={"lr": args.lr, "reg": args.reg})
    trainer = pl.Trainer(
        default_root_dir=os.path.join("logs", model._get_name()),
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        # log_every_n_steps=50, # default
    )
    trainer.fit(model=trainee, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_result = trainer.test(model=trainee, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model=trainee, dataloaders=test_loader, verbose=False)

    # Try to free the memory.
    del trainer
    torch.cuda.empty_cache()

    return {
        "val_acc": val_result[0]["test_acc"],
        "test_acc": test_result[0]["test_acc"],
    }


def plot_logs_from_csv(models):
    plt.style.use("ggplot")

    # Create two figures, one for plotting the training loss, and one for
    # plotting model accuracy during training.
    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()

    for model in models:
        stats = {
            "train_loss":[], "step":[], "val_epoch":[], "train_epoch":[], "val_acc":[], "train_acc":[],
        }
        metrics = os.path.join("logs", model._get_name(), "lightning_logs", "version_0", "metrics.csv")
        with open(metrics, "r") as f:
            metrics_data = csv.reader(f)
            i = 0
            for row in metrics_data:
                # First row. Read the keys of the recorded statistics.
                if i == 0: # skip the first row
                    i += 1
                    assert row == ["train_loss", "epoch", "step", "val_acc", "train_acc", "test_acc"]
                    continue

                # For every other row read the stored stats.
                try: # read loss
                    stats["train_loss"].append(float(row[0]))
                    stats["step"].append(int(row[2]))
                except ValueError:
                    pass
                try: # read train accuracy
                    stats["train_acc"].append(float(row[4]))
                    stats["train_epoch"].append(int(row[1]))
                except ValueError:
                    pass
                try: # read val accuracy
                    stats["val_acc"].append(float(row[3]))
                    stats["val_epoch"].append(int(row[1]))
                except ValueError:
                    pass

        ax_loss.plot(stats["step"], stats["train_loss"], lw=0.8, label=model._get_name())
        ax_acc.plot(stats["train_epoch"], stats["train_acc"], lw=1., label=f"{model._get_name()}_train")
        ax_acc.plot(stats["val_epoch"], stats["val_acc"], ls=":", lw=2.5, label=f"{model._get_name()}_val")

    ax_loss.set_title("Loss value during training")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    fig_loss.savefig(os.path.join("logs", "train_loss.png"))
    plt.close(fig_loss)

    ax_acc.set_title("Accuracy during training")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    fig_acc.savefig(os.path.join("logs", "train_accuracy.png"))
    plt.close(fig_acc)

    return stats


def main(args):
    pl.seed_everything(args.seed)
    models = [GoogLeNet_CIFAR10, ResNet_CIFAR10, DenseNet_CIFAR10]

    # Train the models and store the performance.
    table = []
    for model in models:
        res = train(model, args)
        table.append({
            "model": model._get_name(),
            "val_acc": res["val_acc"],
            "test_acc": res["test_acc"],
            "num_params": sum(len(p.ravel()) for p in model.parameters()),
        })

    # Save the table with model performances.
    with open(os.path.join("logs", "results.json"), "w") as f:
        json.dump(table, f, indent=2)

    # Plot model training stats.
    plot_logs_from_csv(models)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--reg", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--clip_grad", default=None, type=float)
    args = parser.parse_args()

    main(args)

#