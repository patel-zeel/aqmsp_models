from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from astra.torch.models import MLPRegressor, SIRENRegressor

from joblib import Parallel, delayed


class DeepTime(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, repr_dim, dropout):
        super().__init__()
        # self.mlp = SIRENRegressor(x_dim, hidden_dims, repr_dim, dropout=dropout)
        self.mlp = MLPRegressor(x_dim, hidden_dims, repr_dim, dropout=dropout)
        self.log_noise_var = nn.Parameter(torch.tensor(np.log(0.01)))

    def forward(self, x_context, y_context, valid_idx, x_target):
        context_repr = self.mlp(x_context)
        target_repr = self.mlp(x_target)

        context_repr = torch.where(valid_idx, context_repr, 0.0)
        y_context = torch.where(valid_idx, y_context, 0.0)

        cov = context_repr.T @ context_repr
        cov.diagonal().add_(torch.exp(self.log_noise_var))
        xty = context_repr.T @ y_context
        chol = torch.linalg.cholesky(cov)
        w = torch.cholesky_solve(xty, chol)

        y_pred = target_repr @ w
        return y_pred


def fit(train_data, config):
    torch.manual_seed(config["random_state"])
    n_timestamps = len(train_data.time)
    lat_min = train_data["lat"].min().item()
    lat_max = train_data["lat"].max().item()
    lon_min = train_data["lon"].min().item()
    lon_max = train_data["lon"].max().item()

    meta_dict = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        # "mean_y": mean_y,
        # "std_y": std_y,
    }

    train_X = train_data.isel(time=0).to_dataframe().reset_index()[["lat", "lon"]]
    train_X["lat"] = (train_X["lat"] - lat_min) / (lat_max - lat_min)
    train_X["lon"] = (train_X["lon"] - lon_min) / (lon_max - lon_min)
    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    train_X = train_X[np.newaxis, ...].repeat(n_timestamps, 1, 1)

    others = sorted(set(config["features"]) - {"lat", "lon"})
    features = []
    valid_idx_list = []
    for feature in others:
        feat_max = train_data[feature].max().item()
        feat_min = train_data[feature].min().item()
        feat_data = (train_data[feature].values - feat_min) / (feat_max - feat_min)
        features.append(torch.tensor(feat_data, dtype=torch.float32)[..., np.newaxis])
        valid_idx_list.append(~features[-1].isnan())
        meta_dict[f"{feature}_min"] = feat_min
        meta_dict[f"{feature}_max"] = feat_max

    train_X = torch.cat([train_X] + features, dim=-1).to(config["device"])

    train_y = torch.tensor(train_data.value.values, dtype=torch.float32).to(config["device"])[..., np.newaxis]
    # train_y = torch.log1p(train_y)
    valid_idx = ~train_y.isnan()
    for valid_idx_ in valid_idx_list:
        valid_idx = valid_idx & valid_idx_.to(config["device"])

    # mean_y = train_y[valid_idx].mean().item()
    # std_y = train_y[valid_idx].std().item()
    # train_y = (train_y - mean_y) / std_y

    train_y[~valid_idx] = 0.0

    context_size = int(0.5 * train_X.shape[1])

    cnp = DeepTime(len(config["features"]), 1, config["hidden_dims"], config["repr_dim"], config["dropout"]).to(
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
        loss = (target_out_clean - target_y) ** 2  # / (context_y_std**2 + 1e-6)
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

    meta_dict["losses"] = losses
    torch.save(
        meta_dict,
        join(config["working_dir"], "metadata.pt"),
    )


def predict(test_data, train_data, config):
    # load meta
    meta = torch.load(join(config["working_dir"], "metadata.pt"))

    # prepare data
    def prepare(data):
        X = data.isel(time=0).to_dataframe().reset_index()[["lat", "lon"]]
        X["lat"] = (X["lat"] - meta["lat_min"]) / (meta["lat_max"] - meta["lat_min"])
        X["lon"] = (X["lon"] - meta["lon_min"]) / (meta["lon_max"] - meta["lon_min"])
        X = torch.tensor(X.values, dtype=torch.float32)
        X = X[np.newaxis, ...].repeat(len(data.time), 1, 1)

        others = sorted(set(config["features"]) - {"lat", "lon"})
        features = []
        valid_idx_list = []
        for feature in others:
            feat_max = meta[f"{feature}_max"]
            feat_min = meta[f"{feature}_min"]
            feat_data = (data[feature].values - feat_min) / (feat_max - feat_min)
            features.append(torch.tensor(feat_data, dtype=torch.float32)[..., np.newaxis])
            valid_idx_list.append(~features[-1].isnan())

        X = torch.cat([X] + features, dim=-1).to(config["device"])
        y = torch.tensor(data.value.values, dtype=torch.float32).to(config["device"])[..., np.newaxis]
        return X, y, valid_idx_list

    train_X, train_y, valid_idx_list = prepare(train_data)
    test_X, _, _ = prepare(test_data)

    # train_y = torch.log1p(train_y)
    valid_idx = ~train_y.isnan()
    for valid_idx_ in valid_idx_list:
        valid_idx = valid_idx & valid_idx_.to(config["device"])

    train_y[~valid_idx] = 0.0

    cnp = DeepTime(len(config["features"]), 1, config["hidden_dims"], config["repr_dim"], config["dropout"]).to(
        config["device"]
    )
    cnp.load_state_dict(torch.load(join(config["working_dir"], "model.pt")))
    cnp.eval()

    def forward(x, y, valid_idx, x_test):
        y_clean = torch.where(valid_idx, y, 0.0)
        y_mean = y_clean.sum(dim=0, keepdim=True) / valid_idx.sum()
        y_clean_for_std = torch.where(valid_idx, y, y_mean)
        y_std = torch.sqrt(((y_clean_for_std - y_mean) ** 2).sum(dim=0, keepdim=True) / valid_idx.sum())
        y = (y - y_mean) / y_std
        y_pred = cnp(x, y, valid_idx, x_test)
        y_pred = y_pred * y_std + y_mean

        # clip negative predictions
        y_pred = torch.where(y_pred < 0, 0.0, y_pred)
        return y_pred

    with torch.no_grad():
        # print(train_X.shape, train_y.shape, valid_idx.shape)
        y_pred = torch.vmap(forward, in_dims=(0, 0, 0, 0), out_dims=0)(train_X, train_y, valid_idx, test_X)
        # y_pred = y_pred * meta["std_y"] + meta["mean_y"]
        # y_pred = torch.expm1(y_pred)
        y_pred = y_pred.cpu().numpy().squeeze()

    test_data["pred"] = (("time", "station"), y_pred)
    save_path = join(config["working_dir"], "predictions.nc")
    test_data.to_netcdf(save_path)
    print(f"saved {config['model']} predictions to {save_path}")


def fit_predict(train_data, test_data, config):
    fit(train_data, config)
    predict(test_data, train_data, config)
