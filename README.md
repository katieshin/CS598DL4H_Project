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

Data from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) is used, specifically `DIAGNOSES_ICD.csv`. INPREM was provided to us by one of the authors, and we implemented the other baseline models in Python. We compared `Code-Level Accuracy` and `Visit-Level Precision` over 30 visits in 5 visit increments.

## Requirements

To install requirements:

```commandline
pip install -r requirements.txt
```

## Training


## Evaluation


## Pre-trained Models


## Results

