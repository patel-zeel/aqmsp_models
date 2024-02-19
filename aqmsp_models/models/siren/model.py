import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F

from astra.torch.models import SIRENRegressor


def fit(train_data, config):
    raise NotImplementedError(
        "'fit' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Not saving the models because they consume too much space i.e. 13 GB for single fold and a year of data.


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    def train_fn(ts):
        train_df = train_data.sel(time=ts).to_dataframe()
        train_df = train_df.dropna(subset=[config.target]).reset_index()
        train_X = torch.tensor(train_df[config.features].values, dtype=torch.float32)
        train_y = torch.tensor(train_df[[config.target]].values, dtype=torch.float32)

        test_df = test_data.sel(time=ts).to_dataframe().reset_index()
        test_X = torch.tensor(test_df[config.features].values, dtype=torch.float32)

        x_min = train_X.min(dim=0, keepdim=True).values
        x_max = train_X.max(dim=0, keepdim=True).values
        train_X = (train_X - x_min) / (x_max - x_min)
        test_X = (test_X - x_min) / (x_max - x_min)

        y_mean = train_y.mean()
        y_std = train_y.std()
        train_y = (train_y - y_mean) / y_std

        model = SIRENRegressor(input_dim=len(config.features), hidden_dims=config.hidden_dims, dropout=config.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        losses = []
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad()
            pred_y = model(train_X)
            loss = F.mse_loss(pred_y, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred_y = model(test_X)
        pred_y = pred_y * y_std + y_mean

        return pred_y.numpy().squeeze()

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.time.values))
    pred_y = np.array(pred_y_list)
    # print(pred_y.shape)
    # print(test_data)
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
