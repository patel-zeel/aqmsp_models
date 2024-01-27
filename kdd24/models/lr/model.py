import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression


def fit(train_data, config):
    raise NotImplementedError(
        "'fit' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Not saving the models because they consume too much space i.e. 13 GB for single fold and a year of data.


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    test_X = test_data.isel(datetime=0).to_dataframe().reset_index()[config["features"]]

    def train_fn(ts):
        train_df = train_data.sel(datetime=ts).to_dataframe()
        train_df = train_df.dropna(subset=["value"]).reset_index()

        lon_max = train_df["lon"].max()
        lon_min = train_df["lon"].min()
        lat_max = train_df["lat"].max()
        lat_min = train_df["lat"].min()

        train_df["lon"] = (train_df["lon"] - lon_min) / (lon_max - lon_min)
        train_df["lat"] = (train_df["lat"] - lat_min) / (lat_max - lat_min)

        mean = train_df["value"].mean()
        std = train_df["value"].std()
        train_df["value"] = (train_df["value"] - mean) / std

        test_X_local = test_X.copy()
        test_X_local["lon"] = (test_X_local["lon"] - lon_min) / (lon_max - lon_min)
        test_X_local["lat"] = (test_X_local["lat"] - lat_min) / (lat_max - lat_min)

        model = LinearRegression(
            fit_intercept=config["fit_intercept"],
        )
        try:
            model.fit(train_df[config["features"]], train_df["value"])
        except ValueError:
            return np.zeros(len(test_X_local)) * np.nan
        pred_y = model.predict(test_X_local)
        return pred_y * std + mean

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.datetime.values))
    pred_y = np.array(pred_y_list)
    test_data["pred"] = (("datetime", "location_id"), pred_y)
    save_path = f"{config['working_dir']}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config['model']} predictions to {save_path}")
