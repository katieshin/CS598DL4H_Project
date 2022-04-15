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
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, label_ranking_average_precision_score, coverage_error, roc_auc_score

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

DATA_PATH = "../CS598DL4H_Project/data"


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
            self.code2idx[code] = idx

        self.idx2category = sorted(unq_cats)
        self.category2idx = {}
        for idx, cat in enumerate(self.idx2category):
            self.category2idx[cat] = idx

        self.x = data_codes
        self.y = data_categories

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

        self.embedding = nn.Embedding(self.num_codes, self.emb_dim)
        # self.rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        # self.rev_rnn = nn.GRU(emb_dim, hidden_size=emb_dim, batch_first=True)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.rev_lstm = nn.LSTM(self.emb_dim, hidden_size=self.emb_dim, batch_first=True)
        self.fc = nn.Linear(self.emb_dim, self.num_categories)
        self.sigmoid = nn.Sigmoid()

    def sum_embeddings_with_mask(self, x, masks):
        x[masks == 0] = 0
        out = torch.sum(x, 2)
        return out

    def get_last_visit(self, hidden_states, masks):
        s_masks = torch.sum(masks, 2)
        s_masks[s_masks > 0] = 1
        z_masks = torch.sum(s_masks, 1)-1
        mask = torch.LongTensor(*hidden_states.shape[:2])
        mask.zero_()
        mask.scatter_(1, z_masks.view(-1, 1), 1)
        hidden_states[mask == 0] = 0
        out = torch.sum(hidden_states, 1)
        return out

    def forward(self, x, masks):
        """
        Arguments:
            x: the diagnosis sequence of shape (batch_size, # visits, # diagnosis codes)
            masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
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

        # # 1. Pass the sequence through the embedding layer;
        # x = self.embedding(x)
        # # 2. Sum the embeddings for each diagnosis code up for a visit of a patient.
        # x = self.sum_embeddings_with_mask(x, masks)
        #
        # # 3. Pass the embeddings through the RNN layer;
        # output, _ = self.lstm(x)
        # # 4. Obtain the hidden state at the last visit.
        # true_h_n = self.get_last_visit(output, masks)
        #
        # rev_x = self.embedding(rev_x)
        # rev_x = self.sum_embeddings_with_mask(rev_x, rev_masks)
        # output, _ = self.rev_lstm(rev_x)
        # true_h_n_rev = self.get_last_visit(output, rev_masks)
        #
        # # 6. Pass the hidden state through the linear and activation layers.
        # logits = self.fc(torch.cat([true_h_n, true_h_n_rev], 1))
        # probs = self.sigmoid(logits)
        # return probs


def collate_fn(data, **kwargs):
    sequences, labels = zip(*data)

    num_patients = len(sequences)

    max_num_visits = kwargs['max_num_visits']
    max_num_codes = kwargs['max_num_codes']
    max_num_categories = kwargs['max_num_categories']
    category2idx = kwargs['category2idx']

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    y = torch.zeros((num_patients, len(category2idx)), dtype=torch.float)

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
            y[i_patient][category2idx[category]] = 1

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    return train_loader, val_loader, test_loader


def visit_level_precision(k, y_true, y_pred):
    # get top k predictions for each patient
    top_k_val, top_k_ind = torch.topk(y_pred, k, dim=1, sorted=False)

    # num = determine which of top k are correct predictions
    mask = torch.zeros(y_pred.shape).scatter_(1, top_k_ind, top_k_val)
    # mask = (mask > 0.5).float()
    mask = (mask > 0).float()
    num = torch.sum(mask * y_true, dim=1)

    # denom = determine which is smaller (k or number of categories in y_true)
    denom = torch.sum(y_true, 1)
    denom[denom > k] = k
    # precision num/denom
    # return avg(precision)
    return torch.mean(num/denom)


def code_level_accuracy(k, y_true, y_pred):
    # get top k predictions for each patient
    top_k_val, top_k_ind = torch.topk(y_pred, k, dim=1, sorted=False)

    # determine which of top k are correct predictions
    pred_mask = torch.zeros(y_pred.shape).scatter_(1, top_k_ind, top_k_val)
    mask = (pred_mask > 0.5).float()
    num = torch.sum(torch.sum(mask * y_true, dim=1))

    # determine number of labels predicted (p > 0.5)
    # denom = torch.sum((pred_mask > 0.5).float(), dim=1)
    denom = torch.sum(torch.sum((y_pred > 0.5).float(), dim=1))
    denom[denom == 0] = 1  # can't divide by zero
    # accuracy = num/denom
    # return avg(accuracy)
    return torch.mean(num/denom)


def eval_model(model, val_loader):
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_hat = model(x, masks)
        # y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    # ovr and ovo appear to give the same results for our model
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    visit_lvl = visit_level_precision(5, y_true, y_pred)
    code_lvl = code_level_accuracy(5, y_true, y_pred)
    return roc_auc, visit_lvl, code_lvl


def train(model, train_loader, val_loader, n_epochs):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    torch.autograd.set_detect_anomaly(True)

    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks)

            # print(y_hat.get_device())
            # print(y.get_device())

            loss = criterion(y_hat, y)
            # loss = Variable(loss, requires_grad=True)
            # loss = loss.detach().to('cpu')
            # print(loss.get_device())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # y_pred_tmp = (y_hat > 0.5).int()
            y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        # print(accuracy_score(y_true, y_pred))
        # print(label_ranking_average_precision_score(y_true, y_pred))
        # print(roc_auc_score(y_true, y_pred, multi_class='ovo', average='micro'))
        # print(roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro'))
        # print(visit_level_precision(5, y_true, y_pred))
        # print(code_level_accuracy(5, y_true, y_pred))
        roc_auc, visit_lvl, code_lvl = eval_model(model, val_loader)
        print('Epoch: {} \t Validation roc_auc: {:.2f}, visit_lvl: {:.4f}, code_lvl: {:.4f}'.format(epoch + 1, roc_auc, visit_lvl, code_lvl))


def test(model, test_loader):
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in test_loader:
        y_hat = model(x, masks)
        # y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')

    visits = []
    codes = []
    for k in [5, 10, 15, 20, 25, 30]:
        visits.append((k, visit_level_precision(k, y_true, y_pred)))
        codes.append((k, code_level_accuracy(k, y_true, y_pred)))
    return roc_auc, visits, codes


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

    model = RNN(len(dataset.idx2code), len(dataset.category2idx), 256)
    print(model)
    train(model, train_loader, val_loader, params['num_epochs'])
    roc_auc, visit_prec, code_acc = test(model, test_loader)
    print('Test roc_auc: {:.2f}'.format(roc_auc))
    visit_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in visit_prec])
    print('Test visit-level precision@k: {}'.format(visit_str))
    code_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in code_acc])
    print('Test code-level accuracy@k: {}'.format(code_str))
