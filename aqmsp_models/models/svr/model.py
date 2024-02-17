import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.svm import SVR


def fit(train_data, config):
    raise NotImplementedError(
        "'fit' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Not saving the models because they consume too much space i.e. 13 GB for single fold and a year of data.


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    test_X = test_data.isel(time=0).to_dataframe().reset_index()[config["features"]]

    def train_fn(ts):
        train_df = train_data.sel(time=ts).to_dataframe()
        train_df = train_df.dropna(subset=["value"]).reset_index()

        model = SVR()
        try:
            model.fit(train_df[config["features"]], train_df["value"])
        except ValueError:
            return np.zeros(len(test_X)) * np.nan
        pred_y = model.predict(test_X)
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.time.values))
    pred_y = np.array(pred_y_list)
    test_data["pred"] = (("time", "station"), pred_y)
    save_path = f"{config['working_dir']}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config['model']} predictions to {save_path}")
