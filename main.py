import os
from os.path import join
import importlib
import argparse
import toml
from time import time
from collections import namedtuple

# Take arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model name")
parser.add_argument("--dataset", type=str, required=True, help="dataset name")
parser.add_argument("--common_config", type=str, required=True, help="common config file name without .toml extension")
parser.add_argument("--model_config", type=str, required=True, help="config file name without .toml extension")
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
config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

# load configs
common_config = toml.load(f"aqmsp_models/{config.common_config}.toml")
model_config = toml.load(f"aqmsp_models/aqmsp_models/models/{config.model}/{config.model_config}.toml")

# set additional configs manually
common_config["working_dir"] = join(
    common_config["root_dir"],
    f"models/{config.common_config}/{config.model}/{config.dataset}/{config.model_config}/fold_{config.fold}",
)
os.makedirs(common_config["working_dir"], exist_ok=True)

# Freeze it with namedtuple
config = {**common_config, **model_config, **vars(config)}
config = namedtuple("Config", config.keys())(*config.values())
print(config)

# main program
if config.mode == "fit":
    init_time = time()

    # load data
    load_train = importlib.import_module(f"aqmsp_models.datasets.{config.dataset}.dataset").load_train
    train_data = load_train(config)

    # fit model
    fit = importlib.import_module(f"aqmsp_models.models.{config.model}.model").fit
    fit(train_data, config)

    train_time = time() - init_time
    print(f"Training time: {train_time/60:.2f} minutes")

elif config.mode == "predict":
    init_time = time()

    # load data
    load_train = importlib.import_module(f"aqmsp_models.datasets.{config.dataset}.dataset").load_train
    load_test = importlib.import_module(f"aqmsp_models.datasets.{config.dataset}.dataset").load_test
    train_data = load_train(config)
    test_data = load_test(config)

    # predict
    predict = importlib.import_module(f"aqmsp_models.models.{config.model}.model").predict
    predict(test_data, train_data, config)

    test_time = time() - init_time
    print(f"Testing time: {test_time/60:.2f} minutes")

elif config.mode == "fit_predict":
    init_time = time()

    # load data
    load_train = importlib.import_module(f"aqmsp_models.datasets.{config.dataset}.dataset").load_train
    load_test = importlib.import_module(f"aqmsp_models.datasets.{config.dataset}.dataset").load_test
    train_data = load_train(config)
    test_data = load_test(config)

    # fit and predict
    fit_predict = importlib.import_module(f"aqmsp_models.models.{config.model}.model").fit_predict
    fit_predict(train_data, test_data, config)

    train_test_time = time() - init_time
    print(f"Training and testing time: {train_test_time/60:.2f} minutes")
