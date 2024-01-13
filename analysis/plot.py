from os.path import join
import argparse
import toml
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--common_config", type=str, required=True, help="common config file name without .toml extension")
parser.add_argument("--model", type=str, required=True, help="model name")
parser.add_argument("--dataset", type=str, required=True, help="dataset name")
parser.add_argument("--model_config", type=str, required=True, help="config file name without .toml extension")
parser.add_argument("--n_folds", type=int, required=True, help="number of folds")
args = parser.parse_args()

# working dir
config = toml.load(f"kdd24/{args.common_config}.toml")
config = {**config, **vars(args)}
models = args.model.split(",")

plt.figure(figsize=(10, 5))
res_dict = {}
for model in models:
    config["working_dir"] = join(
        config["root_dir"], f"models/{args.common_config}/{model}/{args.dataset}/{args.model_config}"
    )
    # print(config)

    preds_list = []
    for fold in range(args.n_folds):
        load_path = join(config["working_dir"], f"fold_{fold}/predictions.nc")
        preds = xr.open_dataset(load_path)
        preds_list.append(preds)

    error = np.abs(preds["pred"] - preds["value"])
    nonnull_idx = ~np.isnan(error.values)
    res_dict[model] = error.values[nonnull_idx].mean()
    error = error.resample(datetime="1W").mean().mean(dim="location_id")
    plt.plot(error["datetime"], error, label=f"{model}")

    # daily_preds = preds["value"].resample(datetime="1D").mean().mean(dim="location_id")
    # plt.plot(daily_preds["datetime"], daily_preds, label=f"value_fold_{fold}", linestyle="--")

plt.xlabel("datetime")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()
save_path = join("/tmp/mae.pdf")
plt.savefig(save_path)
print(f"Saved to {save_path}")

print(pd.Series(res_dict).sort_values().to_markdown())
