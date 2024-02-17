import re
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
config = toml.load(f"aqmsp_models/{args.common_config}.toml")
config = {**config, **vars(args)}
models = args.model.split(",")

plt.figure(figsize=(10, 5))
df = pd.DataFrame(index=models, columns=[f"fold_{i}" for i in range(args.n_folds)])
for model in models:
    config["working_dir"] = join(
        config["root_dir"], f"models/{args.common_config}/{model}/{args.dataset}/{args.model_config}"
    )
    # print(config)

    for fold in [0, 1, 2, 3, 4]:
        try:
            load_path = join(config["working_dir"], f"fold_{fold}/predictions.nc")
            preds = xr.open_dataset(load_path)

            error = np.abs(preds["pred"] - preds["value"])
            nonnull_idx = ~np.isnan(error.values)
            df.loc[model, f"fold_{fold}"] = error.values[nonnull_idx].mean()
            error = error.resample(time="1D").mean().mean(dim="station")
            # plt.plot(error["time"], error, label=f"{model}")
        except Exception as e:
            # print(e)
            continue

    # daily_preds = preds["value"].resample(time="1D").mean().mean(dim="station")
    # plt.plot(daily_preds["time"], daily_preds, label=f"value_fold_{fold}", linestyle="--")

# plt.xlabel("time")
# plt.ylabel("MAE")
# plt.title("Mean Absolute Error")
# plt.legend()
# save_path = join("/tmp/mae.pdf")
# plt.savefig(save_path)
# print(f"Saved to {save_path}")
df["mean"] = df.mean(axis=1)
df = df.sort_values(by="mean")
print(df)
# rename index
new_index = {
    "zcnp": "Conditional Neural Process",
    "deeptime": "DeepSpace",
    "rf": "Random Forest",
    "idw": "Inverse Distance Weighting",
    "mean": "MeanOfData",
    "1nn": "1-Nearest Neighbor",
    "svr": "Support Vector Regression",
    "lr": "Linear Regression",
    "kriging": "Kriging",
    "gpr": "Gaussian Process Regression",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "spline": "Spline",
}

df = df.rename(index=new_index)

# print(df.style.format("{:.2f}").to_latex(hrules=True))
# add highlight min to the above command
latex = df.style.format("{:.2f}").highlight_min(axis=0).to_latex(hrules=True)
# replace a number followed by "\background-coloryellow" with "\textbf{number}"
latex = re.sub(r"\\background-coloryellow (\d+.\d+)", r"\\textbf{\1}", latex)
print(latex)

# plot last column of df as a bar plot where each bar is of a different color and legend is as per the index
df["mean"].plot(kind="bar", color=["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive"])
# plt.legend()
plt.ylabel("Mean Absolute Error (Lower is Better)")
# include xlabel which is going out of the margin
plt.tight_layout()
plt.savefig("/tmp/mae_bar.pdf")
print(f"Saved to /tmp/mae_bar.pdf")
