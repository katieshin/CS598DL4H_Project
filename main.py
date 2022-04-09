import os
import pandas as pd
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
DATA_PATH = "./data"

from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self):
        # compile data
        df = pd.read_csv(os.path.join(DATA_PATH, 'DIAGNOSES_ICD.csv'))
        data_dict = dict()
        for row in df.to_dict('records'):
            try:
                patient = int(row['SUBJECT_ID'])
                visit = int(row['HADM_ID'])
                code = (int(row['SEQ_NUM']), str(row['ICD9_CODE']))  # allows for sorting
            except Exception as e:
                # ignore rows that have any missing data
                continue

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if visit not in data_dict[patient].keys():
                data_dict[patient][visit] = list()
            data_dict[patient][visit].append(code)

        data = list()
        for i_patient, patient in enumerate(data_dict.keys()):
            patient_list = list()
            for j_visit, visit in enumerate(data_dict[patient].keys()):
                codes = [code for seq, code in sorted(data_dict[patient][visit])]
                patient_list.append(codes)
            data.append(patient_list)

        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def collate_fn(data):
    num_patients = len(data)
    num_visits = [len(patient) for patient in data]
    num_codes = [len(visit) for patient in data for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)

    for i_patient, patient in enumerate(data):
        num_visits = len(patient)
        for j_visit, visit in enumerate(patient):
            for k_code, code in enumerate(visit):
                x[i_patient][j_visit][k_code] = code
                masks[i_patient][j_visit][k_code] = 1
                rev_x[i_patient][num_visits - 1 - j_visit][k_code] = code
                rev_masks[i_patient][num_visits - 1 - j_visit][k_code] = 1

    return x, masks, rev_x, rev_masks


def split_dataset(dataset):
    split_train = int(len(dataset) * 0.75)
    split_val = int(len(dataset) * 0.10)

    lengths = [split_train, split_val, len(dataset) - split_train - split_val]
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)
    print("Length of train dataset:", len(train_dataset))
    print("Length of val dataset:", len(val_dataset))
    print("Length of test dataset:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def load_data(train_dataset, val_dataset, test_dataset, collate_fn):
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, n_epochs):
    pass


def test():
    pass


if __name__ == '__main__':
    dataset = CustomDataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_loader, val_loader, test_loader = load_data(train_dataset, val_dataset, test_dataset, collate_fn)
