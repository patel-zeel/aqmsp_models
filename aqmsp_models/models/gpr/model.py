import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    PairwiseKernel,
    Product,
    Sum,
    Exponentiation,
)


def fit(train_data, config):
    raise NotImplementedError(
        "'fit' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Not saving the models because they consume too much space i.e. 13 GB for single fold and a year of data.


def predict(test_data, train_data, config):
    raise NotImplementedError(
        "'predict' mode is not implemented for 'rf' model. Please use 'fit_predict' mode instead."
    )  # Models are not saved so can not be loaded directly in the predict mode.


def fit_predict(train_data, test_data, config):
    train_X = train_data.isel(time=0).to_dataframe().reset_index()[config.features]
    lat_min = train_X["lat"].min()
    lat_max = train_X["lat"].max()
    lon_min = train_X["lon"].min()
    lon_max = train_X["lon"].max()

    test_X = test_data.isel(time=0).to_dataframe().reset_index()[config.features]
    test_X["lat"] = (test_X["lat"] - lat_min) / (lat_max - lat_min)
    test_X["lon"] = (test_X["lon"] - lon_min) / (lon_max - lon_min)

    def train_fn(ts):
        train_df = train_data.sel(time=ts).to_dataframe()
        train_df["lat"] = (train_df["lat"] - lat_min) / (lat_max - lat_min)
        train_df["lon"] = (train_df["lon"] - lon_min) / (lon_max - lon_min)
        train_df = train_df.dropna(subset=[config.target]).reset_index()

        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.1, 0.9], nu=1.5)  # + WhiteKernel(noise_level=0.1)

        model = GaussianProcessRegressor(
            kernel=kernel, alpha=0.1, random_state=0, normalize_y=True, n_restarts_optimizer=10
        )
        model.fit(train_df[config.features], train_df[config.target])
        pred_y = model.predict(test_X)
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.time.values))
    pred_y = np.array(pred_y_list)
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
