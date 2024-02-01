import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            n_jobs=1,
            random_state=config["random_state"],
            max_depth=config["max_depth"],
        )
        model.fit(train_df[config["features"]], train_df["value"])
        pred_y = model.predict(test_X)
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.datetime.values))
    pred_y = np.array(pred_y_list)
    test_data["pred"] = (("datetime", "location_id"), pred_y)
    save_path = f"{config['working_dir']}/predictions.nc"
    test_data.to_netcdf(save_path)
    print(f"saved {config['model']} predictions to {save_path}")

def fit_predict_grid(train_data, test_data, config):
    test_X = test_data.isel(datetime=0).to_dataframe().reset_index()[config["features"]]

    def train_fn(ts):
        train_df = train_data.sel(datetime=ts).to_dataframe()
        train_df = train_df.dropna(subset=["value"]).reset_index()

        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            n_jobs=1,
            random_state=config["random_state"],
            max_depth=config["max_depth"],
        )
        model.fit(train_df[config["features"]], train_df["value"])
        pred_y = model.predict(test_X)
        return pred_y

    pred_y_list = Parallel(n_jobs=48)(delayed(train_fn)(ts) for ts in tqdm(train_data.datetime.values))
    pred_y = np.array(pred_y_list)
    
    test_lat = test_X['lat'].unique()
    test_lon = test_X['lon'].unique()
    test_lat = np.sort(test_lat)
    test_lon = np.sort(test_lon)
    preds = xr.Dataset(coords={'lon': test_lon, 'lat': test_lat})

    for ts_idx in range(pred_y.shape[0]):    
        pred_grid = np.zeros((len(test_lat), len(test_lon)))    
        for i in range(pred_y.shape[1]):

            lat_idx = np.where(test_lat == test_X['lat'].iloc[i])[0][0] 
            lon_idx = np.where(test_lon == test_X['lon'].iloc[i])[0][0]

            pred_grid[lat_idx, lon_idx] = pred_y[ts_idx, i]

        # Add prediction grid for current timestamp to Dataset
        preds[train_data.datetime[ts_idx].values] = (['lat', 'lon'], pred_grid)
    
    # Set lat, lon as coordinates  
    preds = preds.set_coords(['lat', 'lon'])
    ts = train_data.datetime[0].values
    print(preds[ts])
    preds[ts].plot()
    plt.savefig('prediction_grid.png')

    fig, ax = plt.subplots(figsize=(12, 6))

    cax = preds[ts].plot()

    def animate(frame):
        time_stamp = train_data.datetime[frame].values
        cax.set_array(preds[time_stamp].values.flatten())
        ax.set_title("Time =" + str(time_stamp))
    
    ani = FuncAnimation(fig, animate, frames=len(train_data.datetime), interval=200)

    ani.save('prediction_animation.mp4')