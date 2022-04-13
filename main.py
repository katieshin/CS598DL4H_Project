import json
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import torch.nn as nn

from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

DATA_PATH = ".\\data"


class CustomDataset(Dataset):
    def __init__(self):

        with open(os.path.join(DATA_PATH, 'icd9_map.json')) as fp:
            icd9_map = json.load(fp)

        # compile data
        unq_codes = set()
        unq_cats = set()

        data_dict = dict()
        for f in ['DIAGNOSES_ICD.csv']:
            df = pd.read_csv(os.path.join(DATA_PATH, f), dtype=str)
            for row in df.to_dict('records'):
                try:
                    patient = int(row['SUBJECT_ID'])
                    visit = int(row['HADM_ID'])
                    code_set = (int(row['SEQ_NUM']), row['ICD9_CODE'])  # allows for sorting
                except Exception as e:
                    continue

                if patient not in data_dict.keys():
                    data_dict[patient] = dict()
                if visit not in data_dict[patient].keys():
                    data_dict[patient][visit] = list()
                data_dict[patient][visit].append(code_set)

        data_codes = list()
        data_categories = list()
        for i_patient, patient in enumerate(data_dict.keys()):
            patient_list = list()
            cat_list = list()
            if len(data_dict[patient].keys()) < 2:
                continue  # filter out patients with less than two visits
            sorted_visits = sorted(data_dict[patient].keys())
            for visit in sorted_visits[:-1]:
                codes = list()
                for seq, code in sorted(data_dict[patient][visit]):
                    unq_codes.add(code)
                    codes.append(code)
                patient_list.append(codes)

            cats = list()
            for seq, code in sorted(data_dict[patient][sorted_visits[-1]]):
                cat = icd9_map[code[:3] if code[0] != 'E' else code[:4]]
                unq_cats.add(cat)
                cats.append(cat)
            cat_list.append(list(set(cats)))

            data_codes.append(patient_list)
            data_categories.append(cat_list)

        num_visits = [len(patient) for patient in data_codes]
        num_codes = [len(visit) for patient in data_codes for visit in patient]
        num_categories = [len(visit) for patient in data_categories for visit in patient]
        self.num_patients = len(data_codes)
        self.max_num_visits = max(num_visits)
        self.max_num_codes = max(num_codes)
        self.max_num_categories = max(num_categories)

        self.idx2code = sorted(unq_codes)
        self.code2idx = {}
        for idx, code in enumerate(self.idx2code):
            self.code2idx[code] = idx+1

        self.idx2category = sorted(unq_cats)
        self.category2idx = {}
        for idx, cat in enumerate(self.idx2category):
            self.category2idx[cat] = idx+1

        self.x = data_codes
        self.y = data_categories

        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class RNN(nn.Module):
    def __init__(self, num_codes, num_categories, emb_dim):
        super().__init__()
        """
        TODO: 
            1. Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
            2. Define the RNN using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
            2. Define the RNN for the reverse direction using `nn.GRU()`;
               Set `hidden_size` to 128. Set `batch_first` to True.
            3. Define the linear layers using `nn.Linear()`; Set `in_features` to 256, and `out_features` to 1.
            4. Define the final activation layer using `nn.Sigmoid().

        Arguments:
            num_codes: total number of diagnosis codes
        """
        self.num_codes = num_codes
        self.num_categories = num_categories
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(num_codes, emb_dim)
        # self.rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        # self.rev_rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        self.lstm = nn.LSTM(emb_dim, hidden_size=emb_dim, batch_first=True)
        self.fc = nn.Linear(emb_dim, num_categories)
        self.sigmoid = nn.Sigmoid()

    def sum_embeddings_with_mask(self, x, masks):
        x[masks == 0] = 0
        out = torch.sum(x, 2)
        return out

    def get_last_visit(self, hidden_states, masks):
        # print(masks.shape, hidden_states.shape)
        try:
            s_masks = torch.sum(masks, 2)
            s_masks[s_masks>0] = 1
            z_masks = torch.sum(s_masks, 1)-1
            mask = torch.LongTensor(*hidden_states.shape[:2])
            mask.zero_()
            mask.scatter_(1, z_masks.view(-1, 1), 1)
            hidden_states[mask == 0] = 0
            out = torch.sum(hidden_states, 1)
        except Exception as e:
            # print(hidden_states)
            # torch.set_printoptions(profile="full")
            s_masks = torch.sum(masks, 2)
            # print(masks[12])
            s_masks[s_masks>0] = 1
            # print(s_masks[12])
            z_masks = torch.sum(s_masks, 1)-1
            # print(z_masks)

            # print(masks.shape, hidden_states.shape)
            raise e

        return out

    def forward(self, x, masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        batch_size = x.shape[0]

        # print(self.num_codes, x.shape, masks.shape)

        # 1. Pass the sequence through the embedding layer;
        x = self.embedding(x)
        # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        x = self.sum_embeddings_with_mask(x, masks)

        # 3. Pass the embeddings through the RNN layer;
        output, _ = self.lstm(x)
        logits = self.fc(self.get_last_visit(output, masks))
        probs = self.sigmoid(logits)
        # print(probs.shape)
        return probs


def collate_fn(data, **kwargs):
    sequences, labels = zip(*data)

    num_patients = len(sequences)

    max_num_visits = kwargs['max_num_visits']
    max_num_codes = kwargs['max_num_codes']
    max_num_categories = kwargs['max_num_categories']

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    y = torch.zeros((num_patients, max_num_categories), dtype=torch.long)

    torch.set_printoptions(profile="full")
    for i_patient, patient in enumerate(sequences):
        num_visits = len(patient)
        for j_visit, visit in enumerate(patient):
            for k_code, code in enumerate(visit):
                x[i_patient][j_visit][k_code] = kwargs['code2idx'][code]
                masks[i_patient][j_visit][k_code] = 1
                rev_x[i_patient][num_visits - 1 - j_visit][k_code] = kwargs['code2idx'][code]
                rev_masks[i_patient][num_visits - 1 - j_visit][k_code] = 1

    for i_patient, patient in enumerate(labels):
        for k_code, category in enumerate(patient[-1]):
            y[i_patient][k_code] = kwargs['category2idx'][category]

    return x, masks, rev_x, rev_masks, y


def split_dataset(dataset):
    split_train = int(len(dataset) * 0.75)
    split_val = int(len(dataset) * 0.10)

    lengths = [split_train, split_val, len(dataset) - split_train - split_val]
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)
    print("Length of train dataset:", len(train_dataset))
    print("Length of val dataset:", len(val_dataset))
    print("Length of test dataset:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def load_data(train_dataset, val_dataset, test_dataset, collate_fn, **kwargs):
    batch_size = kwargs['batch_size']
    collate = partial(collate_fn, *[], **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
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
            y_hat = model(x, masks)

            loss = criterion(y_hat, y.float())
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

    params['num_patients'] = dataset.num_patients
    params['max_num_visits'] = dataset.max_num_visits
    params['max_num_codes'] = dataset.max_num_codes
    params['max_num_categories'] = dataset.max_num_categories
    params['idx2code'] = dataset.idx2code
    params['code2idx'] = dataset.code2idx
    params['idx2category'] = dataset.idx2category
    params['category2idx'] = dataset.category2idx

    train_loader, val_loader, test_loader = load_data(train_dataset, val_dataset, test_dataset, collate_fn, **params)

    # model = nn.RNN(dataset.max_num_codes, 128, num_layers=2, dropout=0.5, nonlinearity='tanh', bidirectional=True, batch_first=True)
    model = RNN(len(dataset.idx2code), dataset.max_num_categories, 128)
    print(model)
    train(model, train_loader, val_loader, params['num_epochs'])
