from os.path import join
import numpy as np
import xarray as xr

# period = slice("2021-06-01", "2021-06-01")
period = slice("2020", "2023")


def load(config, mode):
    data_path = join(config["root_dir"], "data/purpleair/lov/data.nc")
    locs_path = join(config["root_dir"], f"data/purpleair/lov/{mode}_{config['fold']}.npy")

    data = xr.open_dataset(data_path)
    locs = np.load(locs_path)

    return data.sel(location_id=locs).sel(datetime=period)


def load_train(config):
    return load(config, "train")


def load_test(config):
    return load(config, "test")
