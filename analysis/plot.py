from os.path import join
import argparse
import toml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model name")
parser.add_argument("--dataset", type=str, required=True, help="dataset name")
parser.add_argument("--config", type=str, required=True, help="config file name without .toml extension")
parser.add_argument("--n_folds", type=int, required=True, help="number of folds")
args = parser.parse_args()

# load configs
common_config = toml.load("kdd24/config.toml")
config = toml.load(f"kdd24/models/{args.model}/{args.config}.toml")
config = {**common_config, **config, **vars(args)}

# working dir
config["working_dir"] = join(config["root_dir"], f"models/{args.model}/{args.dataset}/{args.config}")

preds_list = []
for fold in range(args.n_folds):
    load_path = join(config["root_dir"], f"models/{args.model}/{args.dataset}/{args.config}/fold_{fold}/predictions.nc")
    preds = xr.open_dataset(load_path)

    preds_list.append(preds)

preds = xr.concat(preds_list, dim="location_id")

error = np.abs(preds["pred"] - preds["value"])

# resample to daily
error = error.resample(datetime="1M").mean().mean(dim="location_id")

# plot
plt.figure(figsize=(10, 5))
plt.plot(error["datetime"], error, label=args.model)
plt.xlabel("datetime")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()
save_path = join(config["working_dir"], "mae.pdf")
plt.savefig(save_path)
print(f"Saved to {save_path}")
