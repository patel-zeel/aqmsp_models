import os
from os.path import join
import importlib
import argparse
import toml
from time import time

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model name")
parser.add_argument("--dataset", type=str, required=True, help="dataset name")
parser.add_argument("--config", type=str, required=True, help="config file name without .toml extension")
parser.add_argument("--fold", type=int, required=True, help="fold number")
parser.add_argument(
    "--mode",
    type=str,
    choices=["fit", "predict", "fit_predict"],
    help="""
    'fit' mode: fit and save model and auxiliary files.
    'predict' mode: load model and auxiliary files. Predict and save predictions. 
    'fit_predict' mode: perform fit and predict in a single run.
    """,
    required=True,
)
parser.add_argument("--gpu", type=int, required=False, help="Physical GPU ID")
# take mode argument for train, test. Provide choices
args = parser.parse_args()
print(args, type(args))

# load configs
common_config = toml.load("kdd24/config.toml")
config = toml.load(f"kdd24/models/{args.model}/{args.config}.toml")
config = {**common_config, **config, **vars(args)}
print(config)

# set additional configs manually
config["working_dir"] = join(config["root_dir"], f"models/{args.model}/{args.dataset}/{args.config}/fold_{args.fold}")
os.makedirs(config["working_dir"], exist_ok=True)

# main program
if args.mode == "fit":
    init_time = time()

    # load data
    load_train = importlib.import_module(f"kdd24.datasets.{args.dataset}.dataset").load_train
    train_data = load_train(config)

    # fit model
    fit = importlib.import_module(f"kdd24.models.{args.model}.model").fit
    fit(train_data, config)

    train_time = time() - init_time
    print(f"Training time: {train_time/60:.2f} minutes")

elif args.mode == "predict":
    init_time = time()

    # load data
    load_test = importlib.import_module(f"kdd24.datasets.{args.dataset}.dataset").load_test
    test_data = load_test(config)

    # predict
    predict = importlib.import_module(f"kdd24.models.{args.model}.model").predict
    predict(test_data, config)

    test_time = time() - init_time
    print(f"Testing time: {test_time/60:.2f} minutes")

elif args.mode == "fit_predict":
    init_time = time()

    # load data
    load_train = importlib.import_module(f"kdd24.datasets.{args.dataset}.dataset").load_train
    load_test = importlib.import_module(f"kdd24.datasets.{args.dataset}.dataset").load_test
    train_data = load_train(config)
    test_data = load_test(config)

    # fit and predict
    fit_predict = importlib.import_module(f"kdd24.models.{args.model}.model").fit_predict
    fit_predict(train_data, test_data, config)

    train_test_time = time() - init_time
    print(f"Training and testing time: {train_test_time/60:.2f} minutes")
