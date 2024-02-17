# aqmsp_models

[![CI](https://github.com/patel-zeel/aqmsp_models/actions/workflows/CI.yml/badge.svg)](https://github.com/patel-zeel/aqmsp_models/actions/workflows/CI.yml)

## Experiments
 
```py
python main.py --common_config <common_config_name> --model <model_name> --model_config <config_name> --dataset <dataset_name> --fold <fold_number> --mode <mode> --gpu <physical_gpu_id>
```

| Argument | Options | Description |
| --- | --- | --- |
| common_config | config1, ... | Common config to use. It contains details like features to use.|
| model | rf, ... | Model to use |
| model_config | config1, ... | Model configuration to use |
| dataset | pa_lov, ... | Dataset to use |
| fold | 0, 1, ... | Fold number |
| mode | fit, predict, fit_predict | Fit, predict or fit and predict together |
| gpu | 0, 1, ... | Physical GPU ID (Optional)|


Example:

```py
python main.py --common_config config1 --model rf --model_config config1 --dataset pa_lov --fold 0 --mode fit_predict
```

One can define such commands in a shell script and run them in parallel. For example,

```sh
# measure time
start=`date +%s`

python main.py --model rf --dataset pa_lov --common_config config1 --model_config config2 --mode fit_predict --fold 0
python main.py --model rf --dataset pa_lov --common_config config1 --model_config config2 --mode fit_predict --fold 1
python main.py --model rf --dataset pa_lov --common_config config1 --model_config config2 --mode fit_predict --fold 2
python main.py --model rf --dataset pa_lov --common_config config1 --model_config config2 --mode fit_predict --fold 3
python main.py --model rf --dataset pa_lov --common_config config1 --model_config config2 --mode fit_predict --fold 4

# measure time
end=`date +%s`

runtime=$((end-start))

# print time in miniutes upto 2 decimal places
echo "Total time : $(echo "scale=2; $runtime/60" | bc -l) minutes"
```

> Note that sh files are by default .gitignored. So, you can create any custom `run.sh` file in the root of the repo and run it.

## Reproducibility of the pipeline

- [To Add] A notebook to prepare the data which goes into the pipeline. Assigned to Z in one of the issues.
- Use `notebooks/prepare_lov.ipynb` to prepare the data
- Then use `main.py` to run the experiments


## Repo Structure

```
aqmsp_models
|---models          # for all models
    |---<model1_name>
        |---model.py
        |---config1.toml
        |---config2.toml
        |---...
    |---<model2_name>
        |---model.py
        |---config1.toml
        |---config2.toml
        |---...
    ...
|---datasets        # for all datasets
    |---<dataset1_name>
        |---dataset.py
    |---<dataset2_name>
        |---dataset.py
    ...
main.py             # for running experiments
```

### `model.py` structure

```py
def fit(train_data, config):
    # Fit and save model and auxillary information

def predict(test_data, train_data, config):
    # Predict and save predictions

def fit_predict(train_data, test_data, config):
    # Fit and predict and save predictions. Useful for small models whis do not take much time to fit. For other models, one can use fit and predict separately.
```

## For co-authors

* Open VSCode in the root of the repo
* Install this repo as an editable package
    ```sh
    pip install -e .
    ```
* Define your own model in `models/<model_name>/model.py` which should have `fit`, `predict` and `fit_predict` functions.

* All models must receive the data in same format.
* All models must save their results in the same format.
