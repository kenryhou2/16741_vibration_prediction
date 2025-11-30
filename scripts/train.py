import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO:(ray) Add LSTM model
class VibrationModel(nn.Module):
    def __init__(self, input_size, hidden_dim=32, n_layers=2, output_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size

        layer_sizes = [input_size] + [hidden_dim] * (n_layers - 1)
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class VibrationDataset(torch.utils.data.Dataset):
    def __init__(self, xy):
        # Assumes the last 3 columns are the xyz vibrations
        self.inputs = xy[:, :-3]
        self.labels = xy[:, -3:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx, :]
        label = self.labels[idx, :]
        return x, label


def train_model(bz=16, hidden_dim=16, n_layers=2, n_epochs=100, normalize=True):
    trajs = [np.load(f'../data/cylinder{i}/data_cylinder{i}_11292025.npz') for i in range(2, 4)]

    # TODO: Make it so you can select the different inputs
    # Have to do the sliding window here
    xys = []
    for traj in trajs:
        Xy = np.concatenate([traj['ref_position'][:-1], traj['ref_position'][1:]
                             - traj['tcp_position'][1:]], axis=1)
        Xy = Xy.astype(np.float32)
        xys.append(Xy)
    Xy = np.concatenate(xys, axis=0)

    if normalize:
        xy_mean = Xy.mean(axis=0)
        xy_std = Xy.std(axis=0) + 1e-15
        Xy = (Xy - xy_mean) / xy_std
        # print(Xy.mean(axis=0))
        # print(Xy.std(axis=0))
    np.random.seed(10)
    shuffled_Xy = np.random.permutation(Xy)

    last_train_idx = int(0.9 * Xy.shape[0])
    train_set = shuffled_Xy[:last_train_idx, :]
    val_set = shuffled_Xy[last_train_idx:, :]
    # last_val_idx = int(0.1 * Xy.shape[0] + last_train_idx)
    # test_set = shuffled_Xy[last_val_idx:, :]

    # Load datasets
    train_dataset = VibrationDataset(train_set)
    val_dataset = VibrationDataset(val_set)
    # test_dataset = VibrationDataset(test_set)

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=bz, shuffle=True)
    # test_dl = DataLoader(test_dataset, batch_size=bz, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Model
    output_size = 3
    input_size = Xy.shape[1] - output_size
    model = VibrationModel(input_size, hidden_dim=hidden_dim,
                           n_layers=n_layers, output_size=output_size)

    # Train
    model, train_losses, val_losses = train(
        model, train_dl, val_dl, device=device, n_epochs=n_epochs)

    # Evaluation and plots
    plot_losses(train_losses, val_losses)

    xys = []
    for i in range(2, 4):
        traj = np.load(f'../data/cylinder{i}/data_cylinder{i}_11292025.npz')
        Xy = np.concatenate([traj['ref_position'][:-1], traj['ref_position'][1:]
                             - traj['tcp_position'][1:]], axis=1)
        Xy = Xy.astype(np.float32)
        if normalize:
            Xy = (Xy - xy_mean) / xy_std
        test_whole = VibrationDataset(Xy)
        test_whole_dl = DataLoader(test_whole, batch_size=bz, shuffle=True)
        test_loss, preds = test(model, test_whole_dl, device)
        plot_test_pred(traj, preds, xy_mean, xy_std, traj_i=i)


def train(
    model : nn.Module,
    train_dl : DataLoader,
    val_dl : DataLoader,
    device : str,
    n_epochs: int = 100,
):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for idx, (X, y) in enumerate(train_dl):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            output = loss_fn(pred, y)
            output.backward()
            optimizer.step()
            train_loss += output.item()
        train_losses.append(train_loss / (idx + 1))
        model.eval()
        for idx, (X, y) in enumerate(val_dl):
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(X)
                output = loss_fn(pred, y)
                val_loss += output.item()
        val_losses.append(val_loss / (idx + 1))
    loss_array = np.vstack([train_losses, val_losses]).T
    loss_df = pd.DataFrame(loss_array, columns=['train_loss', 'val_loss'])
    loss_df.to_csv("losses.csv")
    torch.save(model.state_dict(), "saved_model.pt")

    return model, train_losses, val_losses


def test(model, test_dl, device):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    loss = 0
    preds = []
    for idx, (X, y) in enumerate(test_dl):
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(X)
            output = loss_fn(pred, y)
            loss += output.item()
            preds.append(pred.cpu().numpy())
    loss /= idx + 1
    preds = np.concatenate(preds, axis=0)

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


def plot_test_pred(trajs, pred, xy_mean, xy_std, traj_i=0):
    time = trajs['time_rel'][1:]
    ref_traj = trajs['ref_position'][1:]
    tcp_traj = trajs['tcp_position'][1:]
    assert pred.shape == ref_traj.shape, "Different shape ref and pred"
    pred_unnormed = (pred * xy_std[-3:][None, -1]) + xy_mean[-3:][None, -1]
    pred_tcp_traj = ref_traj + pred_unnormed
    fig, axs = plt.subplots(nrows=3, ncols=2)
    subplot_labels = ['x', 'y', 'z']
    for i in range(3):
        pos_ax = axs[i][0]
        pos_ax.plot(time, ref_traj[:, i], label='ref_traj', c='b')
        pos_ax.plot(time, tcp_traj[:, i], label='tcp_traj', c='r')
        pos_ax.plot(time, pred_tcp_traj[:, i], label='pred_tcp_traj', c='g')
        pos_ax.set_xlabel("Time")
        pos_ax.set_ylabel(subplot_labels[i] + " [m]")
        noise_ax = axs[i][1]
        noise_ax.plot(time, tcp_traj[:, i] - ref_traj[:, i], label='actual vibration', c='b')
        noise_ax.plot(time, pred_unnormed[:, i], label='predicted vibration', c='g')
        noise_ax.set_xlabel("Time")
        noise_ax.set_ylabel(subplot_labels[i] + " diff [m]")
    plt.savefig(f'./predicted_plot_traj_{traj_i}.png')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.train:
        model = train_model(bz=args.bz, hidden_dim=args.hidden_dim,
                            n_layers=args.layers, n_epochs=args.n_epochs)
    if args.test:
        # TODO:(ray) Create dataloader for test
        test_dataloader = None
        test(model, test_dataloader, device)


# TODO:(ray) Add loading previous model for test
# Add test dataset arg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True)
    parser.add_argument("--bz", default=16, type=int)
    parser.add_argument("--layers", default=16, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--test", default=False)
    args = parser.parse_args()
    main(args)
