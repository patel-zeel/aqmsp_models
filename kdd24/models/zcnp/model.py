import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed


class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, repr_dim):
        super().__init__()
        self.input = nn.Linear(x_dim + y_dim, hidden_dims[0])

        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.output = nn.Linear(hidden_dims[-1], repr_dim)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.output(x)

        x = x.nanmean(dim=0, keepdim=True)

        return x


class Decoder(nn.Module):
    def __init__(self, repr_dim, x_dim, y_dim, hidden_dims):
        super().__init__()

        self.input = nn.Linear(repr_dim + x_dim, hidden_dims[0])

        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.output = nn.Linear(hidden_dims[-1], y_dim)

    def forward(self, z, x):
        z = z.repeat(x.shape[0], 1)
        x = torch.cat([z, x], dim=1)

        x = F.relu(self.input(x))

        for layer in self.hidden:
            x = F.relu(layer(x))

        x = self.output(x)

        return x


class CNP(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, repr_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, y_dim, hidden_dims, repr_dim)
        self.decoder = Decoder(repr_dim, x_dim, y_dim, hidden_dims)

    def forward(self, x_context, y_context, x_target):
        z = self.encoder(x_context, y_context)
        y_pred = self.decoder(z, x_target)

        return y_pred


def fit(train_data, config):
    train_X = train_data.isel(datetime=0).to_dataframe().reset_index()[config["features"]]
    lat_min = train_X["lat"].min()
    lat_max = train_X["lat"].max()
    lon_min = train_X["lon"].min()
    lon_max = train_X["lon"].max()

    train_X["lat"] = (train_X["lat"] - lat_min) / (lat_max - lat_min)
    train_X["lon"] = (train_X["lon"] - lon_min) / (lon_max - lon_min)
    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    
    train_y = train_data.value.values
    mean_y = train_y.mean()
    std_y = train_y.std()
    train_y = (train_y - mean_y) / std_y
    
    n_timestamps = len(train_data.datetime)
    context_size = int(0.2 * len(train_X))
    
    
    
    def forward(y):
        


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    fit(train_data, config)
    predict(test_data, train_data, config)
