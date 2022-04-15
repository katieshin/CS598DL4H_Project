import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn

from custom_dataset import CustomDataset
from inprem.Loss import UncertaintyLoss
from inprem.doctor.model import Inprem
from inprem.main import args
from rnn import RNN
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, label_ranking_average_precision_score, coverage_error, roc_auc_score

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


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
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    criterion = UncertaintyLoss(opts.task, opts.monto_carlo_for_aleatoric, 2)

    torch.autograd.set_detect_anomaly(True)

    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    # y_pred = torch.FloatTensor()
    # y_true = torch.FloatTensor()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks)

            # REPLACE if not using mac: loss = criterion(y_hat, y.cuda())
            loss = criterion(y_hat, y.to(device))

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    # RNN
    # model = RNN(len(dataset.idx2code), len(dataset.category2idx), 256)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.nn.DataParallel(model).to(device)
    # # REPLACE if not using mac: model = torch.nn.DataParallel(model).cuda()
    # print(model)
    # train(model, train_loader, val_loader, params['num_epochs'])
    # roc_auc, visit_prec, code_acc = test(model, test_loader)
    # print('Test roc_auc: {:.2f}'.format(roc_auc))
    # visit_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in visit_prec])
    # print('Test visit-level precision@k: {}'.format(visit_str))
    # code_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in code_acc])
    # print('Test code-level accuracy@k: {}'.format(code_str))

    # INPREM
    train_set_max_visit = 0
    train_length = len(train_dataset)
    for i in range(train_length):
        num_visits = len(train_dataset[i][0])
        if num_visits > train_set_max_visit:
            train_set_max_visit = num_visits

    valid_set_max_visit = 0
    valid_length = len(val_dataset)
    for i in range(valid_length):
        num_visits = len(val_dataset[i][0])
        if num_visits > valid_set_max_visit:
            valid_set_max_visit = num_visits

    test_set_max_visit = 0
    test_length = len(test_dataset)
    for i in range(test_length):
        num_visits = len(test_dataset[i][0])
        if num_visits > test_set_max_visit:
            test_set_max_visit = num_visits

    opts = args().parse_args()
    max_visits = max(train_set_max_visit, valid_set_max_visit, test_set_max_visit)
    input_dim = max_visits - 2
    net = Inprem(opts.task, input_dim, 2, opts.emb_dim, max_visits,
                 opts.n_depth, opts.n_head, opts.d_k, opts.d_v, opts.d_inner,
                 opts.cap_uncertainty, opts.drop_rate, False, opts.dp, opts.dvp, opts.ds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.nn.DataParallel(net).to(device)
    # REPLACE if not using mac: model = torch.nn.DataParallel(model).cuda()
    train(net, train_loader, val_loader, params['num_epochs'])
    roc_auc, visit_prec, code_acc = test(net, test_loader)
    print('Test roc_auc: {:.2f}'.format(roc_auc))
    visit_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in visit_prec])
    print('Test visit-level precision@k: {}'.format(visit_str))
    code_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in code_acc])
    print('Test code-level accuracy@k: {}'.format(code_str))
