import argparse

import matplotlib.pylot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class VibrationModel(nn.Module):
    def __init__(self, input_size, hidden_dim=32, n_layers=2, output_size=3):
        super().__init__()
        layer_sizes = [input_size] + [hidden_dim] * (n_layers - 1)
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_size))

        self.network = nn.Sequential(layers)

    def forward(self, x):
        return self.network(x)


# TODO:(ray) Create dataloader for vibration trajectory data
class VibrationDataset(torch.utils.data.Dataset):
    def __init__(self, target_trajs, npzs, sliding_window):
        self.labels = np.zeros(0)

    def __len__(self):
        return len(self.labels)

    # TODO:(ray) fix get item
    def __getitem__(self, idx):
        pass
        # return img, label


def train_model(bz=16, hidden_dim=16, n_layers=2):

    # Load datasets
    train_dataset = VibrationDataset()
    val_dataset = VibrationDataset()
    test_dataset = VibrationDataset()

    # Create DataLoaders
    bz = 1
    train_dl = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=bz, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=bz, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Model
    input_size = 0
    output_size = 0
    model = VibrationModel(input_size, output_size=output_size)

    # Train
    model, train_losses, val_losses = train(
        model, train_dl, val_dl, device=device)

    # Evaluation and plots
    plot_losses(train_losses, val_losses)
    test_loss = test(model, test_dl, device)
    print(test_loss)


def train(
    model : nn.Module,
    train_dl : DataLoader,
    val_dl : DataLoader,
    device : str,
):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    n_epochs = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for idx, (imgs, labels) in enumerate(train_dl):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            output = loss_fn(pred, labels)
            output.backward()
            optimizer.step()
            train_loss += output.item()
        train_losses.append(train_loss / (idx + 1))
        model.eval()
        for idx, (imgs, labels) in enumerate(val_dl):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred = model(imgs)
                output = loss_fn(pred, labels)
                val_loss += output.item()
        val_losses.append(val_loss / (idx + 1))
    loss_array = np.vstack([train_losses, val_losses]).T
    loss_df = pd.DataFrame(loss_array, columns=['train_loss', 'val_loss'])
    loss_df.to_csv("losses.csv")
    torch.save(model.state_dict(), "saved_model.pt")

    return model, train_losses, val_losses


def test(model, test_dl, device):
    model.eval()
    loss = []
    preds = []
    for idx, (X, y) in enumerate(test_dl):
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.numpy())

    return loss, preds


def plot_losses(train_loss, val_loss):
    fig, ax1 = plt.subplots(1, 1)
    ax1 : plt.Axes
    epochs = np.arange(len(train_loss))
    ax1.plot(epochs, train_loss, label='Train Loss')
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training plot")
    plt.legend()
    fig.tight_layout()
    plt.savefig("training_plot.png")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.train:
        model = train_model(bz=args.bz)
    if args.test:
        # TODO:(ray) Create dataloader for test
        test_dataloader = None
        test(model, test_dataloader, device)


# TODO:(ray) Add loading previous model for test
# Add test dataset arg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True)
    parser.add_argument("--bz", default=16)
    parser.add_argument("--layers", default=16)
    parser.add_argument("--hidden_dim", default=16)
    parser.add_argument("--test", default=True)
    args = parser.parse_args()
    main(args)
