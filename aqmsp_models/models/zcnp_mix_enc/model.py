from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed


class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, repr_dim, dropout):
        super().__init__()
        self.dropout = dropout

        self.input = nn.Linear(x_dim + y_dim, hidden_dims[0])

        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.output = nn.Linear(hidden_dims[-1], repr_dim)

        self.attention_net = nn.Linear(x_dim, 1)

    def forward(self, x, y, valid_idx, x_target):
        out = torch.cat([x, y], dim=-1)
        out = F.relu(self.input(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        for layer in self.hidden:
            out = F.relu(layer(out))
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.output(out)

        w_enc = self.attention_net(x)
        w_dec = self.attention_net(x_target)
        w = w_enc @ w_dec.T
        w = F.softmax(w.sum(dim=1, keepdim=True), dim=0)
        out = w * out

        out = torch.where(valid_idx, out, 0.0)

        out = out.sum(dim=0, keepdim=True) / valid_idx.sum()

        return out


class Decoder(nn.Module):
    def __init__(self, repr_dim, x_dim, y_dim, hidden_dims, dropout):
        super().__init__()
        self.dropout = dropout

        self.input = nn.Linear(repr_dim + x_dim, hidden_dims[0])

        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.output = nn.Linear(hidden_dims[-1], y_dim)

    def forward(self, z, x):
        z = z.repeat(x.shape[0], 1)
        x = torch.cat([z, x], dim=1)

        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.hidden:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output(x)

        return x


class GatedCNP(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, repr_dim, dropout, n_encoders):
        super().__init__()
        self.n_encoders = n_encoders

        self.encoder = Encoder(x_dim, y_dim, hidden_dims, repr_dim, dropout)
        # for i in range(self.n_encoders):
        #     setattr(self, f"decoder_{i}", Decoder(repr_dim, x_dim, y_dim, hidden_dims, dropout))
        self.decoder = Decoder(repr_dim, x_dim, y_dim, hidden_dims, dropout)

        # self.merger = nn.Linear(x_dim, 1)

    def forward(self, x_context, y_context, y_context_valid, x_target):
        # z_list = []
        # for i in range(self.n_encoders):
        #     z_enc = getattr(self, f"encoder_{i}")(x_context, y_context, y_context_valid)
        #     z_list.append(z_enc)
        z = self.encoder(x_context, y_context, y_context_valid, x_target)
        y_pred = self.decoder(z, x_target)

        # w_enc = self.merger(x_context)
        # w_dec = self.merger(x_target)
        # w = w_enc
        # w = F.softmax(w, dim=1, keepdim=True)
        # z = torch.cat(z_list, dim=0)
        # print(z.shape, w.shape, "------------------")
        # z_mix = w @ z
        # y_pred_list = []
        # for i in range(self.n_encoders):
        #     decoder = getattr(self, f"decoder_{i}")
        #     y_pred = decoder(z, x_target)
        #     y_pred_list.append(y_pred)
        # y_pred = torch.cat(y_pred_list, dim=1)

        # y_pred =

        return y_pred


def fit(train_data, config):
    torch.manual_seed(config["random_state"])
    n_timestamps = len(train_data.time)
    train_X = train_data.isel(time=0).to_dataframe().reset_index()[config.features]
    lat_min = train_X["lat"].min().item()
    lat_max = train_X["lat"].max().item()
    lon_min = train_X["lon"].min().item()
    lon_max = train_X["lon"].max().item()

    train_X["lat"] = (train_X["lat"] - lat_min) / (lat_max - lat_min)
    train_X["lon"] = (train_X["lon"] - lon_min) / (lon_max - lon_min)
    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    ####### Temporarily
    train_X = train_X[np.newaxis, ...].repeat(n_timestamps, 1, 1).to(config["device"])

    train_y = torch.tensor(train_data.value.values, dtype=torch.float32).to(config["device"])[..., np.newaxis]
    # train_y = torch.log1p(train_y)
    valid_idx = ~train_y.isnan()
    # mean_y = train_y[valid_idx].mean().item()
    # std_y = train_y[valid_idx].std().item()
    # train_y = (train_y - mean_y) / std_y

    train_y[~valid_idx] = 0.0

    context_size = int(0.5 * train_X.shape[1])

    cnp = GatedCNP(2, 1, config["hidden_dims"], config["repr_dim"], config["dropout"], config["n_encoders"]).to(
        config["device"]
    )

    def loss_fn(x, y, valid_idx):
        idx = torch.randperm(len(y))
        context_idx = idx[:context_size]
        target_idx = idx[context_size:]
        context_x = x[context_idx]
        context_y = y[context_idx]
        context_valid = valid_idx[context_idx]
        target_x = x[target_idx]
        target_y = y[target_idx]
        target_valid = valid_idx[target_idx]

        # scale
        context_y_mean = context_y.sum(dim=0, keepdim=True) / context_valid.sum()
        context_y_for_std = torch.where(context_valid, context_y, context_y_mean)
        context_y_std = torch.sqrt(
            ((context_y_for_std - context_y_mean) ** 2).sum(dim=0, keepdim=True) / context_valid.sum()
        )
        context_y = (context_y - context_y_mean) / context_y_std

        target_out = cnp(context_x, context_y, context_valid, target_x) * context_y_std + context_y_mean
        target_out_clean = torch.where(target_valid, target_out, 0.0)
        loss = (target_out_clean - target_y) ** 2
        # print(loss.isnan().sum(), target_valid.sum())
        # print(loss.isnan().sum(), target_valid.sum())
        # print(len(target_valid), target_valid, target_valid.sum())
        return loss.sum() / target_valid.sum()

    epochs = config["epochs"]
    pbar = tqdm(range(epochs))
    vloss = torch.vmap(loss_fn, randomness="different")
    optimizer = torch.optim.Adam(cnp.parameters(), lr=config["lr"])
    losses = []

    for epoch in pbar:
        optimizer.zero_grad()
        loss = vloss(train_X, train_y, valid_idx).mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

    torch.save(cnp.state_dict(), join(config["working_dir"], "model.pt"))
    torch.save(
        {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            # "mean_y": mean_y,
            # "std_y": std_y,
        },
        join(config["working_dir"], "metadata.pt"),
    )


def predict(test_data, train_data, config):
    # load meta
    meta = torch.load(join(config["working_dir"], "metadata.pt"))

    # prepare data
    train_X = train_data.isel(time=0).to_dataframe().reset_index()[config.features]
    train_X["lat"] = (train_X["lat"] - meta["lat_min"]) / (meta["lat_max"] - meta["lat_min"])
    train_X["lon"] = (train_X["lon"] - meta["lon_min"]) / (meta["lon_max"] - meta["lon_min"])
    train_X = torch.tensor(train_X.values, dtype=torch.float32)

    ####### Temporarily
    train_X = train_X[np.newaxis, ...].repeat(len(train_data.time), 1, 1).to(config["device"])

    test_X = test_data.isel(time=0).to_dataframe().reset_index()[config.features]
    test_X["lat"] = (test_X["lat"] - meta["lat_min"]) / (meta["lat_max"] - meta["lat_min"])
    test_X["lon"] = (test_X["lon"] - meta["lon_min"]) / (meta["lon_max"] - meta["lon_min"])
    test_X = torch.tensor(test_X.values, dtype=torch.float32).to(config["device"])

    train_y = torch.tensor(train_data.value.values, dtype=torch.float32).to(config["device"])[..., np.newaxis]
    valid_idx = ~train_y.isnan()

    ####### Temporarily
    # test_X = test_X[np.newaxis, ...].repeat(len(test_data.time), 1, 1).to(config["device"])

    cnp = GatedCNP(2, 1, config["hidden_dims"], config["repr_dim"], config["dropout"], config["n_encoders"]).to(
        config["device"]
    )
    cnp.load_state_dict(torch.load(join(config["working_dir"], "model.pt")))
    cnp.eval()

    def forward(x, y, valid_idx):
        y_clean = torch.where(valid_idx, y, 0.0)
        y_mean = y_clean.sum(dim=0, keepdim=True) / valid_idx.sum()
        y_clean_for_std = torch.where(valid_idx, y, y_mean)
        y_std = torch.sqrt(((y_clean_for_std - y_mean) ** 2).sum(dim=0, keepdim=True) / valid_idx.sum())
        y = (y - y_mean) / y_std
        y_pred = cnp(x, y, valid_idx, test_X)
        return y_pred * y_std + y_mean

    with torch.no_grad():
        # print(train_X.shape, train_y.shape, valid_idx.shape)
        y_pred = torch.vmap(forward, in_dims=(0, 0, 0), out_dims=0)(train_X, train_y, valid_idx)
        # y_pred = y_pred * meta["std_y"] + meta["mean_y"]
        # y_pred = torch.expm1(y_pred)
        y_pred = y_pred.cpu().numpy().squeeze()

    test_data[f"{config.target}_pred"] = (("time", "station"), y_pred)
    save_path = join(config["working_dir"], "predictions.nc")
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")


def fit_predict(train_data, test_data, config):
    fit(train_data, config)
    predict(test_data, train_data, config)
