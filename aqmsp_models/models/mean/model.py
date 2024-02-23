import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor


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
        test_df = test_data.sel(time=ts).to_dataframe()

        model = DummyRegressor(strategy="mean")

        model.fit(train_df[["lat", "lon"]], train_df[config.target])
        pred_y = model.predict(test_df[["lat", "lon"]])
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.time.values))
    pred_y = np.array(pred_y_list)
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
