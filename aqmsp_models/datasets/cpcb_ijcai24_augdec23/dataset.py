from os.path import join
import numpy as np
import xarray as xr


def load(config, mode):
    data_path = join("/home/patel_zeel/aqmsp/aqmsp_data/datasets/cpcb/ijcai24/data.nc")
    locs_path = join(config.root_dir, f"data/purpleair/lov/{mode}_{config.fold}.npy")
    locs_path = join(f"/home/patel_zeel/aqmsp/aqmsp_data/datasets/cpcb/ijcai24/fold_{config.fold}_{mode}.npy")

    print("Opening data")
    data = xr.open_dataset(data_path).sel(time=slice("2023-08", "2023-12"))
    locs = np.load(locs_path, allow_pickle=True)

    print("Selecting stations")
    return data.sel(station=locs)


def load_train(config):
    return load(config, mode="train")


def load_test(config):
    return load(config, mode="test")
