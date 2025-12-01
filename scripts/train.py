import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class VibrationFFModel(nn.Module):
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


# TODO:(ray) Add LSTM model
class VibrationLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=2, output_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


class VibrationDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        # Assumes the last 3 columns are the xyz vibrations
        self.inputs = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx, :]
        label = self.labels[idx, :]
        return x, label


def create_sequences(X, Y, seq_len=1, only_last=False):
    xs, ys = [], []
    for i in range(len(Y) - seq_len):
        x = X[i:i + seq_len, ...]
        if only_last:
            y = Y[i + seq_len, ...]
        else:
            y = Y[i + 1:i + seq_len + 1, ...]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)


def train_model(use_lstm=True,
                bz=16, hidden_dim=16,
                n_layers=2, n_epochs=100,
                normalize=False, seq_len=1, lr=1e-3):
    # TODO: Use multiple trajectories
    trajs = [np.load(glob(f'../data/cylinder{i}/data_cylinder{i}*.npz')[0]) for i in range(1, 2)]

    Xs = []
    Ys = []
    means = []
    stds = []
    for traj in trajs:
        X = (np.concatenate([traj['ref_position']], axis=1)).astype(
            np.float32)
        Y = (traj['tcp_position'] - traj['ref_position']).astype(np.float32)
        # TODO:(ray) Not sure if should normalize in the trajectory or overall
        if normalize:
            x_mean = X.mean(axis=0)
            y_mean = Y.mean(axis=0)
            x_std = X.std(axis=0)
            y_std = Y.std(axis=0)
            means.append((x_mean, y_mean))
            stds.append((x_std, y_std))
            X = (X - x_mean) / x_std
            Y = (Y - y_mean) / y_std
        if use_lstm:
            xs, ys = create_sequences(X, Y, seq_len=seq_len)
        else:
            xs, ys = create_sequences(X, Y, seq_len=1)
        Xs.append(xs)
        Ys.append(ys)

    X = torch.concatenate(Xs, axis=0)
    Y = torch.concatenate(Ys, axis=0)

    # NOTE: Should not shuffle for time series
    # shuffled_Xy = np.random.permutation(Xy)

    last_train_idx = int(0.8 * X.shape[0])
    last_val_idx = int(0.1 * X.shape[0] + last_train_idx)
    train_X, train_y = X[:last_train_idx, :], Y[:last_train_idx, :]
    val_X, val_y = X[last_train_idx:last_val_idx, :], Y[last_train_idx:last_val_idx, :]
    test_X, test_y = X[last_val_idx:, :], Y[last_val_idx:, :]

    # Load datasets
    train_dataset = VibrationDataset(train_X, train_y)
    val_dataset = VibrationDataset(val_X, val_y)
    test_dataset = VibrationDataset(test_X, test_y)

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=bz, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=bz, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Model
    input_size = X.shape[-1]
    output_size = Y.shape[-1]
    if use_lstm:
        model = VibrationLSTMModel(input_size, hidden_dim=hidden_dim,
                                   n_layers=n_layers, output_size=output_size)
    else:
        model = VibrationFFModel(input_size, hidden_dim=hidden_dim,
                                 n_layers=n_layers, output_size=output_size)

    # Train
    model, train_losses, val_losses = train(
        model, train_dl, val_dl, device=device, n_epochs=n_epochs, lr=lr)

    # Evaluation and plots
    plot_losses(train_losses, val_losses)
    if not normalize:
        y_std = 1
        y_mean = 0

    with torch.no_grad():
        y_train_pred = model(train_X.to(device))
        y_train_pred = y_train_pred[:, -1, ...].cpu().numpy() * y_std + y_mean
        y_val_pred = model(val_X.to(device))
        y_val_pred = y_val_pred[:, -1, ...].cpu().numpy() * y_std + y_mean
        y_test_pred = model(test_X.to(device))
        y_test_pred = y_test_pred[:, -1, ...].cpu().numpy() * y_std + y_mean

    plot_test_pred(trajs[0], y_train_pred, y_val_pred, y_test_pred, seq_len=seq_len)


def train(
    model : nn.Module,
    train_dl : DataLoader,
    val_dl : DataLoader,
    device : str,
    n_epochs: int = 100,
    lr: float = 1e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    for epoch in tqdm(range(1, n_epochs + 1)):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for idx, (X_batch, y_batch) in enumerate(train_dl):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            output = loss_fn(pred, y_batch)
            output.backward()
            optimizer.step()
            train_loss += output.item()
        train_losses.append(train_loss / (idx + 1))
        model.eval()
        for idx, (X_batch, y_batch) in enumerate(val_dl):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                pred = model(X_batch)
                output = loss_fn(pred, y_batch)
                val_loss += output.item()
            if output.item() < best_val_loss:
                best_val_loss = best_val_loss
        val_losses.append(val_loss / (idx + 1))
        if epoch % 100 == 0:
            print(f"Epoch {epoch} train loss {train_losses[-1]} val loss {val_losses[-1]}")
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


def plot_test_pred(trajs, y_train_pred, y_val_pred, y_test_pred, seq_len=1, traj_i=0):
    time = trajs['time_rel']
    ref_traj = trajs['ref_position']
    tcp_traj = trajs['tcp_position']
    last_train_ix = len(y_train_pred) + seq_len
    last_val_ix = len(y_val_pred) + last_train_ix
    pred_train_tcp_traj = ref_traj[seq_len:last_train_ix, ...] + y_train_pred
    pred_val_tcp_traj = ref_traj[last_train_ix:last_val_ix, ...] + y_val_pred
    pred_test_tcp_traj = ref_traj[last_val_ix:, ...] + y_test_pred
    fig, axs = plt.subplots(nrows=3, ncols=2)
    subplot_labels = ['x', 'y', 'z']
    for i in range(0, 3):
        pos_ax = axs[i][0]
        pos_ax.plot(time, tcp_traj[..., i], label='tcp_traj', c='r')
        pos_ax.plot(time[seq_len:last_train_ix], pred_train_tcp_traj[..., i],
                    label='pred_train_tcp_traj', c='b')
        pos_ax.plot(time[last_train_ix:last_val_ix],
                    pred_val_tcp_traj[..., i], label='pred_val_tcp_traj', c='g')
        pos_ax.plot(time[last_val_ix:], pred_test_tcp_traj[..., i],
                    label='pred_test_tcp_traj', c='m')
        pos_ax.set_xlabel("Time")
        pos_ax.set_ylabel(subplot_labels[i] + " [m]")
        noise_ax = axs[i][1]
        noise_ax.plot(time, (tcp_traj - ref_traj)[..., i], label='actual vibration', c='r')
        noise_ax.plot(time[seq_len:last_train_ix], y_train_pred[..., i],
                      label='pred_train_delta', c='b')
        noise_ax.plot(time[last_train_ix:last_val_ix],
                      y_val_pred[..., i], label='pred_delta', c='g')
        noise_ax.plot(time[last_val_ix:], y_test_pred[..., i],
                      label='pred_test_delta', c='m')
        noise_ax.set_xlabel("Time")
        noise_ax.set_ylabel(subplot_labels[i] + " diff [m]")
    plt.savefig(f'./predicted_plot_traj_{traj_i}.png')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.train:
        model = train_model(use_lstm=args.lstm, bz=args.bz, hidden_dim=args.hidden_dim,
                            n_layers=args.layers, n_epochs=args.n_epochs, normalize=args.normalize, seq_len=args.seq_len, lr=args.lr)
    if args.test:
        # TODO:(ray) Create dataloader for test
        test_dataloader = None
        test(model, test_dataloader, device)


# TODO:(ray) Add loading previous model for test
# Add test dataset arg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Fix arguments
    parser.add_argument("--lstm", action="store_false")
    parser.add_argument("--normalize", action="store_false")
    parser.add_argument("--train", default=True)
    parser.add_argument("--bz", default=16, type=int)
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--seq_len", default=1, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--test", default=False)
    args = parser.parse_args()
    main(args)
