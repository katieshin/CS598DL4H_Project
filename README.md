# CS598DL4H Project: Reproducing INPREM

## Project Details

As part of UIUC CS598 Deep Learning for Healthcare course, we have decided to reproduce [<strong><em>INPREM: An Interpretable and Trustworthy Predictive Model for Healthcare</em></strong>](https://dl-acm-org.proxy2.library.illinois.edu/doi/10.1145/3394486.3403087). The goal of this project is to reproduce the experiment results within the paper and also add an ablation of our choosing to see how it influences the outcome.

### Paper Details

The paper implements the INPREM model and compares it to currently widely-used models, such as:
- CNN
- RNN
- RNN+
- Dipole
- RETAIN

The goal of the INPREM model is to be used for clinical prediction tasks, which current models are not very suitable for. Data from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) is used, specifically `DIAGNOSES_ICD.csv`. INPREM was provided to us by one of the authors, and we implemented the other baseline models in Python. We compared `Code-Level Accuracy` and `Visit-Level Precision` over 30 visits in 5 visit increments.

## Requirements

To install requirements:

```commandline
pip install -r requirements.txt
```

## Training

There is an option to use pre-trained models (available for all models including INPREM above) or train the model from scratch. This is set by specifying an argument when running `main.py`.

The following are available arguments that can be set:

- `--model`: type of model to run
  - default: `INPREM`
  - choices: `CNN, RNN, RETAIN, DIPOLE, RNNplus, INPREM`
- `--emb_dim`: size of medical variable (or code) embedding
  - default: 256
- `--train`: boolean to train the model or use the pre-trained model
  - default: `False`
- `--epochs`: number of iterations
  - default: 25
- `--batch-size`: batch size for data
  - default: 32
- `--drop_rate`: drop-out rate before each weight layer
  - default: 0.5
- `--optimizer`: optimizer for model
  - default: `Adam`
  - choices: `SGD, Adadelta, Adam`
- `--lr`: learning rate for each step
  - default: 5e-4
- `--weight_decay`: weight decay for the model run
  - default: 1e-4
- `--cap_uncertainty`: boolean for capping uncertainty
  - default: `False`
- `--save_model_dir`: directory to save the model with the best performance
  - default: `os.path.join(base_dir, 'saved_models')`
- `--data_csv`: data file which will be used to train and evaluate a model
  - default: `os.path.join(base_dir, 'data', 'DIAGNOSES_ICD.csv')`
- `--icd9map`: location for ICD9 code mapping to categories
  - default: `os.path.join(base_dir, 'data', 'icd9_map.json')`

A sample command that can be run is:
```commandline
python3 main.py --model=CNN --train=True
```

## Evaluation

Running `main.py` will output the following evaluation results:
- ROC AUC (Area Under the Receiver Operating Characteristic Curve)
- Visit-level precision for 30 visits with increments of 5
- Code-level accuracy for 30 iterations with increments of 5
- Time to test the model
- Total time to run

## Results

