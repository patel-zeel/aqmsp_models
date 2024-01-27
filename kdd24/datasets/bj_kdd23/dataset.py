from os.path import join
import numpy as np
import xarray as xr


def load(config, mode):
    data_path = join(config["root_dir"], "data/beijing/bj_kdd23/data.nc")
    locs_path = join(config["root_dir"], f"data/beijing/bj_kdd23/{mode}_{config['fold']}.npy")

    data = xr.open_dataset(data_path)
    locs = np.load(locs_path)

    return data.sel(location_id=locs)


def load_train(config):
    return load(config, "train")


def load_test(config):
    return load(config, "test")
