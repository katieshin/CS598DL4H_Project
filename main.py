import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


class CustomDataset(Dataset):
    def __init__(self):
        # compile data
        unq_codes = set()
        unq_cats = set()

        df = pd.read_csv('DIAGNOSES_ICD.csv')
        data_dict = dict()
        for row in df.to_dict('records'):
            try:
                patient = int(row['SUBJECT_ID'])
                visit = int(row['HADM_ID'])
                code = row['ICD9_CODE']
                if len(code) < 5:
                    code += '0' * (5 - len(code))

                code_set = (int(row['SEQ_NUM']), code)  # allows for sorting
                unq_codes.add(code)
                unq_cats.add(code[0:3])
            except Exception as e:
                # ignore rows that have any missing data
                continue

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if visit not in data_dict[patient].keys():
                data_dict[patient][visit] = list()
            data_dict[patient][visit].append(code_set)

        code_map = dict()
        rev_code_map = dict()
        for idx, code in enumerate(sorted(unq_codes)):
            code_map[code] = idx + 1
            rev_code_map[idx + 1] = code

        data = list()
        for i_patient, patient in enumerate(data_dict.keys()):
            patient_list = list()
            if len(data_dict[patient].keys()) < 2:
                continue  # filter out patients with less than two visits
            for j_visit, visit in enumerate(sorted(data_dict[patient].keys())):
                codes = [code_map[code] for seq, code in sorted(data_dict[patient][visit])]
                patient_list.append(codes)
            data.append(patient_list)

        num_visits = [len(patient) for patient in data]
        num_codes = [len(visit) for patient in data for visit in patient]
        self.num_patients = len(data)
        self.max_num_visits = max(num_visits)
        self.max_num_codes = max(num_codes)
        self.x = data

        print('number of patients', self.num_patients)
        print('number of visits', sum(num_visits))
        print('average number of visits', sum(num_visits) / len(num_visits))
        print('number of unique codes', len(unq_codes))
        print('average number of codes per visit', sum(num_codes) / len(num_codes))
        print('max number of codes per visit', self.max_num_codes)
        print('number of unique categories', len(unq_cats))
        print()
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def collate_fn(data, **kwargs):
    num_patients = len(data)
    # print('kwargs', kwargs)

    max_num_visits = kwargs.get('max_num_visits', max([len(patient) for patient in data]))
    max_num_codes = kwargs.get('max_num_codes', max([len(visit) for patient in data for visit in patient]))

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


def load_data(train_dataset, val_dataset, test_dataset, collate_fn, batch_size, **kwargs):
    # print('loading data')
    # collate = partial(collate_fn, [], kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # TODO: replace
    criterion = nn.BCELoss()  # TODO: replace
    eval_model = None  # TODO: replace

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks, rev_x, rev_masks)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        # p, r, f, roc_auc = eval_model(model, val_loader)
        # print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'
        # .format(epoch + 1, p, r, f, roc_auc))


def test():
    pass


if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'num_epochs': 10
    }

    dataset = CustomDataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    kwargs = {'max_num_visits': dataset.max_num_visits, 'max_num_codes': dataset.max_num_codes}
    train_loader, val_loader, test_loader = load_data(train_dataset, val_dataset, test_dataset,
                                                      collate_fn, params['batch_size'], **kwargs)

    # model = nn.RNN(dataset.max_num_codes, 128, num_layers=2, dropout=0.5, nonlinearity='tanh',
    #                bidirectional=True, batch_first=True)
    # train(model, train_loader, val_loader, params['num_epochs'])
