import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
from sklearn.ensemble import RandomForestRegressor

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

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        n_jobs=-1,
        random_state=config.random_state,
        max_depth=config.max_depth,
    )
    model.fit(train_df[config.features], train_df[config.target])
    # save model
    save_path = f"{config.working_dir}/model.joblib"
    dump(model, save_path)

    def pred_fn(ts):
        model = load(save_path)
        test_X = test_data.sel(time=ts).to_dataframe().reset_index()[config.features]
        pred_y = model.predict(test_X)
        return pred_y

    pred_y_list = Parallel(n_jobs=32)(delayed(pred_fn)(ts) for ts in tqdm(train_data.time.values))
    # pred_y_list = [pred_fn(ts) for ts in tqdm(train_data.time.values)]
    print(len(pred_y_list), "sized pred_y_list")
    pred_y = np.array(pred_y_list)
    test_data[f"{config.target}_pred"] = (("time", "station"), pred_y)
    save_path = f"{config.working_dir}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config.model} predictions to {save_path}")
