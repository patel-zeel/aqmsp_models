import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat

from joblib import Parallel, delayed


class Encoder(nn.Module):
    def __init__(self, n_spatial, n_spatio_temporal, dropout):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = dropout

        self.input_fc = nn.Linear(n_spatial, 100)
        self.input_lstm = nn.LSTM(n_spatio_temporal + 1, 300, num_layers=2, batch_first=True)

        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(
        self, x_spatial, x_spatio_temporal, y_spatio_temporal
    ):  # x_spatial: (batch_size, n_context_stations, n_spatial), x_spatio_temporal: (batch_size, n_context_stations, window_size, n_spatio_temporal), y_spatio_temporal: (batch_size, n_context_stations, window_size, 1)
        batch_size = x_spatial.shape[0]
        xy_spatio_temporal = torch.cat([x_spatio_temporal, y_spatio_temporal], dim=-1)

        z_spatial = torch.vmap(lambda x: F.dropout(self.act(self.input_fc(x)), p=self.dropout), randomness="same")(
            x_spatial
        )

        xy_spatio_temporal = rearrange(
            xy_spatio_temporal,
            "batch_size n_context_stations window_size n_spatio_temporal -> (batch_size n_context_stations) window_size n_spatio_temporal",
        )
        z_spatio_temporal, _ = self.input_lstm(xy_spatio_temporal)
        z_spatio_temporal = rearrange(
            z_spatio_temporal[:, -1, :],
            "(batch_size n_context_stations) lstm_out -> batch_size n_context_stations lstm_out",
            batch_size=batch_size,
        )

        z_concat = torch.cat([z_spatial, z_spatio_temporal], dim=-1)
        z_concat = torch.vmap(lambda x: F.dropout(self.act(self.fc2(x)), p=self.dropout), randomness="same")(z_concat)
        z_concat = torch.vmap(lambda x: self.fc3(x), randomness="same")(z_concat)

        return z_concat


class decoder(nn.Module):
    def __init__(self, n_spatial, n_spatio_temporal, dropout):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = dropout

        self.input_fc = nn.Linear(n_spatial, 100)
        self.input_lstm = nn.LSTM(n_spatio_temporal, 300, num_layers=2, batch_first=True, dropout=self.dropout)

        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(
        self, x_spatial, x_spatio_temporal
    ):  # x_spatial: (batch_size, n_target_stations, n_spatial), x_spatio_temporal: (batch_size, n_target_stations, window_size, n_spatio_temporal), y_spatio_temporal: (batch_size, n_target_stations, window_size, 1)
        # We don't use vmap out of the box because it doesn't support the LSTM layer.
        batch_size = x_spatial.shape[0]

        z_spatial = torch.vmap(lambda x: F.dropout(self.act(self.input_fc(x)), p=self.dropout), randomness="same")(
            x_spatial
        )

        x_spatio_temporal = rearrange(
            x_spatio_temporal,
            "batch_size n_target_stations window_size n_spatio_temporal -> (batch_size n_target_stations) window_size n_spatio_temporal",
        )
        z_spatio_temporal, _ = self.input_lstm(x_spatio_temporal)
        z_spatio_temporal = rearrange(
            z_spatio_temporal[:, -1, :],
            "(batch_size n_target_stations) lstm_out -> batch_size n_target_stations lstm_out",
            batch_size=batch_size,
        )

        z_concat = torch.cat([z_spatial, z_spatio_temporal], dim=-1)
        z_concat = torch.vmap(lambda x: F.dropout(self.act(self.fc2(x)), p=self.dropout), randomness="same")(z_concat)

        return z_concat


class AttentionNet(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(
        self, z_context, z_target
    ):  # z_context: (batch_size, num_context_stations, 200), z_target: (batch_size, num_target_stations, 200)
        num_context_stations = z_context.shape[1]

        def single_forward(z_target, z_context):
            z_target = repeat(
                z_target, "z_dim -> num_content_stations z_dim", num_content_stations=num_context_stations
            )
            z = torch.cat(
                [z_target, z_context], dim=-1
            )  # (num_context_stations, 200) + (num_context_stations, 200) = (num_context_stations, 400)
            z = F.dropout(
                F.relu(self.fc(z)), p=self.dropout
            )  # (num_context_stations, 400) -> (num_context_stations, 200)
            z = self.fc2(z)  # (num_context_stations, 200) -> (num_context_stations, 1)
            z = F.softmax(z, dim=0)  # (num_context_stations, 1)
            return rearrange(z, "num_context_stations 1 -> num_context_stations")

        # (num_context_stations, 200) + (num_target_stations, 200) -> (num_context_stations, num_target_stations)
        multi_forward = torch.vmap(single_forward, in_dims=(0, None), out_dims=1, randomness="same")
        # (batch_size, num_context_stations, 200) + (batch_size, num_target_stations, 200) -> (batch_size, num_context_stations, num_target_stations)
        attention = torch.vmap(multi_forward, randomness="same")(z_target, z_context)
        return attention


class ADAIN(nn.Module):
    def __init__(self, n_spatial, n_spatio_temporal, dropout):
        super().__init__()
        self.dropout = dropout
        self.encoder = Encoder(n_spatial, n_spatio_temporal, dropout)
        self.decoder = decoder(n_spatial, n_spatio_temporal, dropout)
        self.attention = AttentionNet(dropout)
        self.fc = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(
        self,
        x_context_spatial,
        x_context_spatio_temporal,
        y_context_spatio_temporal,
        x_target_spatial,
        x_target_spatio_temporal,
    ):
        # y_mean = y_context_spatio_temporal.mean(dim=(1, 2, 3), keepdim=True)
        # y_std = y_context_spatio_temporal.std(dim=(1, 2, 3), keepdim=True)
        # y_context_spatio_temporal = (y_context_spatio_temporal - y_mean) / y_std

        z_context = self.encoder(
            x_context_spatial, x_context_spatio_temporal, y_context_spatio_temporal
        )  # (batch_size, num_context_stations, 200)
        z_target = self.decoder(x_target_spatial, x_target_spatio_temporal)  # (batch_size, num_target_stations, 200)
        attention = self.attention(z_context, z_target)  # (batch_size, num_context_stations, num_target_stations)

        def get_output(attention, z_target):
            attention = rearrange(attention, "batch_size num_context_stations -> batch_size num_context_stations 1")
            output = attention * z_context  # (batch_size, num_context_stations, 200)
            output = torch.sum(output, dim=1)  # (batch_size, 200)
            fused = torch.cat([output, z_target], dim=-1)  # (batch_size, 400)
            fused = F.dropout(F.relu(self.fc(fused)), p=self.dropout)  # (batch_size, 400) -> (batch_size, 200)
            fused = self.fc2(fused)  # (batch_size, 200) -> (batch_size, 1)
            return fused

        # (batch_size, num_context_stations, num_target_stations)
        output = torch.vmap(get_output, in_dims=(2, 1), out_dims=1, randomness="same")(attention, z_target)
        # print(output.shape)
        return output  # * y_std[:, :, -1, :] + y_mean[:, :, -1, :]


static_features = ["lat", "lon", "elevation", "pop_1km", "pop_2km", "pop_3km"]


def fit(train_data, config):
    torch.manual_seed(config.random_state)

    meta_dict = {}
    for feature in config.features:
        fet_min = train_data[feature].min().item()
        fet_max = train_data[feature].max().item()

        meta_dict.update(
            {
                f"{feature}_min": fet_min,
                f"{feature}_max": fet_max,
            }
        )

        train_data[feature] = (train_data[feature] - fet_min) / (fet_max - fet_min)

    y_mean = train_data[config.target].mean().item()
    y_std = train_data[config.target].std().item()
    meta_dict.update(
        {
            "y_mean": y_mean,
            "y_std": y_std,
        }
    )
    train_data[config.target] = (train_data[config.target] - y_mean) / y_std

    class CustomDataset(Dataset):
        def __init__(self, ds, window_size):
            self.ds = ds
            self.window_size = window_size
            self.ts = self.ds.time.values

        def __len__(self):
            return len(self.ts) - self.window_size

        def __getitem__(self, idx):
            t_past = self.ts[idx : idx + self.window_size]
            t = self.ts[idx + self.window_size]
            given_static_features = [fet for fet in config.features if fet in static_features]
            X_static = self.ds.sel(time=t).to_dataframe()[given_static_features].values
            X_dynamic = np.concatenate(
                [
                    self.ds.sel(time=t_past)[var].values.T[..., None]
                    for var in config.features
                    if var not in static_features
                ],
                axis=-1,
            )
            y_dynamic = self.ds.sel(time=t_past)[config.target].values.T[..., None]

            X_static = torch.tensor(X_static, dtype=torch.float32)
            X_dynamic = torch.tensor(X_dynamic, dtype=torch.float32)
            y_dynamic = torch.tensor(y_dynamic, dtype=torch.float32)
            # print(X_static.shape, X_dynamic.shape, y_dynamic.shape)

            idx = np.random.permutation(len(X_static))
            num_context = int(config.context_fraction * len(X_static))
            X_context_spatial = X_static[idx[:num_context]]
            X_context_spatio_temporal = X_dynamic[idx[:num_context]]
            y_context_spatio_temporal = y_dynamic[idx[:num_context]]
            X_target_spatial = X_static[idx[num_context:]]
            X_target_spatio_temporal = X_dynamic[idx[num_context:]]
            y_target = y_dynamic[idx[num_context:], -1, :]  # take the last value of the window as the target
            return (
                X_context_spatial,
                X_context_spatio_temporal,
                y_context_spatio_temporal,
                X_target_spatial,
                X_target_spatio_temporal,
                y_target,
            )

    dataset = CustomDataset(train_data, config.window_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    given_static_features = [fet for fet in config.features if fet in static_features]

    model = ADAIN(len(given_static_features), len(config.features) - len(given_static_features), config.dropout).to(
        config.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = []
    best_loss = np.inf
    for epoch in range(config.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader)
        for (
            X_context_spatial,
            X_context_spatio_temporal,
            y_context_spatio_temporal,
            X_target_spatial,
            X_target_spatio_temporal,
            y_target,
        ) in pbar:
            X_context_spatial = X_context_spatial.to(config.device)
            X_context_spatio_temporal = X_context_spatio_temporal.to(config.device)
            y_context_spatio_temporal = y_context_spatio_temporal.to(config.device)
            X_target_spatial = X_target_spatial.to(config.device)
            X_target_spatio_temporal = X_target_spatio_temporal.to(config.device)
            y_target = y_target.to(config.device)

            out = model(
                X_context_spatial,
                X_context_spatio_temporal,
                y_context_spatio_temporal,
                X_target_spatial,
                X_target_spatio_temporal,
            )
            loss = F.mse_loss(out, y_target)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

        losses.append(epoch_loss / len(dataloader))

        if losses[-1] < best_loss:
            best_loss = losses[-1]
            torch.save(model.state_dict(), join(config.working_dir, "model.pt"))

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {losses[-1]:.4f}")

    meta_dict["losses"] = losses
    torch.save(
        meta_dict,
        join(config.working_dir, "metadata.pt"),
    )


def predict(test_data, train_data, config):
    # load meta
    meta = torch.load(join(config.working_dir, "metadata.pt"))

    # prepare test data
    for feature in config.features:
        fet_min = meta[f"{feature}_min"]
        fet_max = meta[f"{feature}_max"]
        train_data[feature] = (train_data[feature] - fet_min) / (fet_max - fet_min)
        test_data[feature] = (test_data[feature] - fet_min) / (fet_max - fet_min)

    y_mean = meta["y_mean"]
    y_std = meta["y_std"]
    train_data[config.target] = (train_data[config.target] - y_mean) / y_std

    class CustomDataset(Dataset):
        def __init__(self, train_ds, test_ds, window_size):
            self.train_ds = train_ds
            self.test_ds = test_ds
            self.window_size = window_size
            self.ts = self.test_ds.time.values

        def __len__(self):
            return len(self.ts) - self.window_size

        def __getitem__(self, idx):
            t_past = self.ts[idx : idx + self.window_size]
            t = self.ts[idx + self.window_size]
            given_static_features = [fet for fet in config.features if fet in static_features]
            X_context_static = self.train_ds.sel(time=t).to_dataframe()[given_static_features].values
            X_context_dynamic = np.concatenate(
                [
                    self.train_ds.sel(time=t_past)[var].values.T[..., None]
                    for var in config.features
                    if var not in static_features
                ],
                axis=-1,
            )
            y_context_dynamic = self.train_ds.sel(time=t_past)[config.target].values.T[..., None]

            X_target_spatial = self.test_ds.sel(time=t).to_dataframe()[given_static_features].values
            X_target_dynamic = np.concatenate(
                [
                    self.test_ds.sel(time=t_past)[var].values.T[..., None]
                    for var in config.features
                    if var not in static_features
                ],
                axis=-1,
            )
            y_target = self.test_ds.sel(time=t)[config.target].values.reshape(-1, 1)

            X_context_spatial = torch.tensor(X_context_static, dtype=torch.float32)
            X_context_spatio_temporal = torch.tensor(X_context_dynamic, dtype=torch.float32)
            y_context_spatio_temporal = torch.tensor(y_context_dynamic, dtype=torch.float32)
            X_target_spatial = torch.tensor(X_target_spatial, dtype=torch.float32)
            X_target_spatio_temporal = torch.tensor(X_target_dynamic, dtype=torch.float32)
            y_target = torch.tensor(y_target, dtype=torch.float32)

            return (
                X_context_spatial,
                X_context_spatio_temporal,
                y_context_spatio_temporal,
                X_target_spatial,
                X_target_spatio_temporal,
                y_target,
            )

    # dataset
    dataset = CustomDataset(train_data, test_data, config.window_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # load model
    given_static_features = [fet for fet in config.features if fet in static_features]
    model = ADAIN(len(given_static_features), len(config.features) - len(given_static_features), config.dropout).to(
        config.device
    )
    model.load_state_dict(torch.load(join(config.working_dir, "model.pt")))
    model.eval()

    with torch.no_grad():
        y_pred = []
        for (
            X_context_spatial,
            X_context_spatio_temporal,
            y_context_spatio_temporal,
            X_target_spatial,
            X_target_spatio_temporal,
            y_target,
        ) in tqdm(dataloader):
            pred_y = model(
                X_context_spatial.to(config.device),
                X_context_spatio_temporal.to(config.device),
                y_context_spatio_temporal.to(config.device),
                X_target_spatial.to(config.device),
                X_target_spatio_temporal.to(config.device),
            )
            y_pred.append(pred_y.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = y_pred * y_std + y_mean

    # add nans for initial window
    y_pred = np.concatenate([np.full((config.window_size, y_pred.shape[1], 1), np.nan), y_pred], axis=0)

    test_data[f"{config.target}_pred"] = (("time", "station"), y_pred.squeeze())
    save_path = join(config.working_dir, "predictions.nc")
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")


def fit_predict(train_data, test_data, config):
    fit(train_data, config)
    predict(test_data, train_data, config)
