import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from polire import Kriging


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
        train_df = train_df[train_df[f"{config.target}_missing"] == False]
        X = train_df[["lat", "lon"]].values
        y = train_df[config.target].values
        X_test = test_data.sel(time=ts).to_dataframe()[["lat", "lon"]].values

        model = Kriging(variogram_model=config.variogram_model)
        model.fit(X, y)
        pred_y = model.predict(X_test)
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.time.values))
    pred_y = np.array(pred_y_list)
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
