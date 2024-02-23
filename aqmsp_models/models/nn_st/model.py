import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
from astra.torch.models import MLPRegressor, SIRENRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm


ignore_features = [
    "PM2.5",
    "NO2",
    "SO2",
    "CO",
    "PM2.5_missing",
    "NO2_missing",
    "SO2_missing",
    "CO_missing",
    "hour",
    "dayofweek",
    "dayofyear",
    "is_day",
    "sunday",
    "saturday",
    "PM10",
    "PM10_missing",
    "NH3",
    "NOx",
    "O",
    "Ozone",
    "NO",
    "Benzene",
    "Eth-Benzene",
    "Xylene",
    "MP-Xylene",
    "Toluene",
    "time",
    "lat",
    "lon",
    "pop_1km",
    "pop_2km",
    "pop_3km",
    "pop_4km",
    "pop_5km",
    "pop_10km",
    "station",
    "d_motorway",
    "elevation",
]


def fit(train_data, config):
    raise NotImplementedError(
        "'fit' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Not saving the models because they consume too much space i.e. 13 GB for single fold and a year of data.


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    print("Converting to dataframe")
    train_df = train_data.to_dataframe()
    print("Columns:", train_df.columns)
    train_df = train_df[train_df[f"{config.target}_missing"] == False]

    print("Editing features")
    ## Edit features

    for feature in list(config.features):
        for lag in config.lags:
            if feature not in ignore_features:
                config.features.append(f"{feature}_lag_{lag}")

    print("Fitting with", len(config.features), "features")
    fit_scaler = {}
    for fet in tqdm(config.features):
        fet_max = train_df[fet].max() + 1e-6
        fet_min = train_df[fet].min()
        fit_scaler[fet] = {"max": fet_max, "min": fet_min}
        train_df[fet] = (train_df[fet] - fet_min) / (fet_max - fet_min)

    y_mean = train_df[config.target].mean()
    y_std = train_df[config.target].std()
    train_df[config.target] = (train_df[config.target] - y_mean) / y_std

    # model = MLPRegressor(len(config.features), config.hidden_dims, 1, dropout=config.dropout).to(config.device)
    model = SIRENRegressor(len(config.features), config.hidden_dims, 1, dropout=config.dropout).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        running_loss = 0
        pbar = tqdm(range(0, len(train_df), config.batch_size))
        for i in pbar:
            batch_df = train_df.iloc[i : i + config.batch_size]
            X = batch_df[config.features].values
            y = batch_df[[config.target]].values
            X = torch.tensor(X, dtype=torch.float32).to(config.device)
            y = torch.tensor(y, dtype=torch.float32).to(config.device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"loss: {loss.item():.5f}")
        epoch_loss = running_loss / len(train_df) * config.batch_size
        print(f"Epoch {epoch+1} loss: {epoch_loss:.8f}")

    # save model
    # save_path = f"{config.working_dir}/model.pt"
    # torch.save(model, save_path)

    # print(f"saved {config.model} model to {save_path}")

    # predict
    model.eval()

    for fet in tqdm(config.features):
        fet_max = fit_scaler[fet]["max"]
        fet_min = fit_scaler[fet]["min"]
        test_data[fet] = (test_data[fet] - fet_min) / (fet_max - fet_min)

    with torch.no_grad():
        test_df = test_data.to_dataframe().reset_index()
        pred_y_list = []
        for ts in tqdm(train_data.time.values):
            test_X = test_df[test_df.time == ts][config.features].values
            test_X = torch.tensor(test_X, dtype=torch.float32).to(config.device)
            pred_y = model(test_X).cpu().numpy()
            pred_y_list.append(pred_y.reshape(1, -1))

    pred_y_list = np.concatenate(pred_y_list)
    # pred_y_list = [pred_fn(ts) for ts in tqdm(train_data.time.values)]
    print(len(pred_y_list), "sized pred_y_list", pred_y_list.shape)
    pred_y = pred_y_list
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
