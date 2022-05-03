# CS598DL4H Project: Reproducing INPREM

## Project Details

As part of UIUC CS598 Deep Learning for Healthcare course, we have decided to reproduce [<strong><em>INPREM: An Interpretable and Trustworthy Predictive Model for Healthcare</em></strong>](https://dl-acm-org.proxy2.library.illinois.edu/doi/10.1145/3394486.3403087). The goal of this project is to reproduce the experiment results within the paper and also add an ablation of our choosing to see how it influences the outcome.

### Paper Details

The paper implements the INPREM model (and its variations) and compares it to currently widely-used models, such as:
- CNN
- RNN
- RNN+
- Dipole
- RETAIN

The goal of the INPREM model is to be used for clinical prediction tasks, which current models are not very suitable for. Data from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) is used, specifically `DIAGNOSES_ICD.csv`. INPREM was provided to us by one of the authors, and we implemented the other baseline models in Python. We compared `Code-Level Accuracy` and `Visit-Level Precision` over 30 visits in 5 visit increments.

## Requirements

Create a python 3.8 environment.

To install requirements:

```commandline
pip install -r requirements.txt
```

(Optional) For Windows and Linux users, to enable the models to run on your GPU the appropriate version of the torch must be installed.

```commandline
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
```

## Training

When running `main.py` to train the models you must use the `--train` argument along with the model specification, `--model {model}`. Replace `{model}` with one of the available models. 
The default arguments are what we used to train the model in our experiments.

The following are available arguments that can be set:

- `--model`: type of model to run
  - default: `INPREM`
  - choices: `CNN, RNN, RETAIN, DIPOLE, RNNplus, INPREM, INPREM_b, INPREM_s, INPREM_o`
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
- `--save_model_dir`: directory to save the model with the best performance
  - default: `os.path.join(base_dir, 'saved_models')`
- `--data_csv`: data file which will be used to train and evaluate a model
  - default: `os.path.join(base_dir, 'data', 'DIAGNOSES_ICD.csv')`
- `--icd9map`: location for ICD9 code mapping to categories
  - default: `os.path.join(base_dir, 'data', 'icd9_map.json')`

A sample command that can be run is:
```commandline
python3 main.py --model=RNN --train
```

## Evaluation
All models will run the evaluation step after training is complete.
To evaluate a pre-trained model, run:
```commandline
python3 main.py --model=RNN
```

A model evaluation will result in the following metrics:
- ROC AUC (Area Under the Receiver Operating Characteristic Curve)
- Visit-level precision @ k for k={5, 10, 15, 20, 25, 30}
- Code-level accuracy @ k for k={5, 10, 15, 20, 25, 30}
- Time taken to test the model
- Total time taken to run the entire script


## Pre-trained Models
Pre-trained models are avialable in the saved_models folder of this repository. You can specify a directory to load additional pre-trained models from by specifying the `--save_model_dir` parameter flag.

All models were trained using the default parameter flags.

## Results

<table>
  <tr>
    <th></th>
    <th>Model</th>
    <th>ROC AUC</th>
    <th>Time to<br>Test [sec]</th>
    <th>Time to<br>Run [sec]</th>
  </tr>
  <tr>
    <td rowspan="5">Baselines</td>
    <td>CNN</td>
	<td>0.9083</td>
	<td>2.05</td>
	<td>9.40</td>
  </tr>
  <tr>
    <td>RNN</td>
	<td>0.8992</td>
	<td>1.62</td>
	<td>9.27</td>
  </tr>
  <tr>
    <td>RNN+</td>
	<td>0.9142</td>
	<td>1.58</td>
	<td>9.21</td>
  </tr>
  <tr>
    <td>RETAIN</td>
	<td>0.8843</td>
	<td>1.62</td>
	<td>9.22</td>
  </tr>
  <tr>
    <td>Dipole</td>
	<td>0.9009</td>
	<td>1.63</td>
	<td>9.25</td>
  </tr>
  <tr>
    <td rowspan="4">INPREM</td>
    <td>INPREM</td>
	<td>0.4918</td>
	<td>2.87</td>
	<td>10.31</td>
  </tr>
  <tr>
    <td>INPREM<sub>b-</sub></td>
	<td>0.6051</td>
	<td>3.15</td>
	<td>10.66</td>
  </tr>
  <tr>
    <td>INPREM<sub>o-</sub></td>
	<td>0.5879</td>
	<td>3.03</td>
	<td>10.84</td>
  </tr>
  <tr>
    <td>INPREM<sub>s-</sub></td>
	<td>0.6053</td>
	<td>2.16</td>
	<td>9.93</td>
  </tr>
</table>

<table>
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Model</th>
    <th colspan="6">Code-Level Accuracy@k</th>
    <th colspan="6">Visit-Level Precision@k</th>
  </tr>
  <tr>
    <td>5</td>
    <td>10</td>
    <td>15</td>
    <td>20</td>
    <td>25</td>
    <td>30</td>
    <td>5</td>
    <td>10</td>
    <td>15</td>
    <td>20</td>
    <td>25</td>
    <td>30</td>
  </tr>
  <tr>
    <td rowspan="5">Baselines</td>
    <td>CNN</td>
    <td>0.5266</td>
    <td>0.5736</td>
    <td>0.5736</td>
    <td>0.5736</td>
    <td>0.5736</td>
    <td>0.5736</td>
    <td>0.5280</td>
    <td>0.5279</td>
    <td>0.5869</td>
    <td>0.6638</td>
    <td>0.7320</td>
    <td>0.7779</td>
  </tr>
  <tr>
    <td>RNN</td><td>0.4155</td>
	<td>0.5363</td>
	<td>0.5567</td>
	<td>0.5582</td>
	<td>0.5582</td>
	<td>0.5582</td>
	<td>0.6089</td>
	<td>0.5758</td>
	<td>0.6246</td>
	<td>0.6874</td>
	<td>0.7426</td>
	<td>0.7834</td>
  </tr>
  <tr>
    <td>RNN+</td>
	<td>0.6238</td>
	<td>0.7094</td>
	<td>0.7123</td>
	<td>0.7123</td>
	<td>0.7123</td>
	<td>0.7123</td>
	<td>0.6300</td>
	<td>0.5931</td>
	<td>0.6409</td>
	<td>0.7053</td>
	<td>0.7630</td>
	<td>0.8071</td>
  </tr>
  <tr>
    <td>RETAIN</td>
	<td>0.5601</td>
	<td>0.6481</td>
	<td>0.6557</td>
	<td>0.6557</td>
	<td>0.6557</td>
	<td>0.6557</td>
	<td>0.6009</td>
	<td>0.5661</td>
	<td>0.6198</td>
	<td>0.6858</td>
	<td>0.7482</td>
	<td>0.7919</td>
  </tr>
  <tr>
    <td>Dipole</td>
	<td>0.5635</td>
	<td>0.5635</td>
	<td>0.5635</td>
	<td>0.5635</td>
	<td>0.5635</td>
	<td>0.5635</td>
	<td>0.4772</td>
	<td>0.4756</td>
	<td>0.5442</td>
	<td>0.6348</td>
	<td>0.7028</td>
	<td>0.7539</td>
  </tr>
  <tr>
    <td rowspan="4">INPREM</td>
    <td>INPREM</td>
	<td>0.0249</td>
	<td>0.0249</td>
	<td>0.0249</td>
	<td>0.0249</td>
	<td>0.0249</td>
	<td>0.0249</td>
	<td>0.0102</td>
	<td>0.0065</td>
	<td>0.0059</td>
	<td>0.0065</td>
	<td>0.0067</td>
	<td>0.0079</td>
  </tr>
  <tr>
    <td>INPREM<sub>b-</sub></td>
	<td>0.4815</td>
	<td>0.5143</td>
	<td>0.5143</td>
	<td>0.5143</td>
	<td>0.5143</td>
	<td>0.5143</td>
	<td>0.4146</td>
	<td>0.2629</td>
	<td>0.2321</td>
	<td>0.2270</td>
	<td>0.2264</td>
	<td>0.2263</td>
  </tr>
  <tr>
    <td>INPREM<sub>o-</sub></td>
	<td>0.5393</td>
	<td>0.5409</td>
	<td>0.5409</td>
	<td>0.5409</td>
	<td>0.5409</td>
	<td>0.5409</td>
	<td>0.3689</td>
	<td>0.2228</td>
	<td>0.1987</td>
	<td>0.1948</td>
	<td>0.1943</td>
	<td>0.1945</td>
  </tr>
  <tr>
    <td>INPREM<sub>s-</sub></td>
	<td>0.4817</td>
	<td>0.5185</td>
	<td>0.5185</td>
	<td>0.5185</td>
	<td>0.5185</td>
	<td>0.5185</td>
	<td>0.4117</td>
	<td>0.2641</td>
	<td>0.2337</td>
	<td>0.2286</td>
	<td>0.2280</td>
	<td>0.2279</td>
  </tr>
</table>