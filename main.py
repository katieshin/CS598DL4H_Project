import argparse
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from baseline_models import CNN, RNN, RNNplus, RETAIN, DIPOLE
from custom_dataset import CustomDataset
from inprem import Inprem, UncertaintyLoss

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
base_dir = os.path.dirname(__file__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('CNN', 'RNN', 'RETAIN', 'DIPOLE', 'RNNplus', 'INPREM'), help='Choose the optimizer.', default='INPREM')
    parser.add_argument('--emb_dim', type=int, default=256, help='The size of medical variable (or code) embedding.')
    parser.add_argument('--train', action='store_true', help='Weather capture uncertainty.', default=False)
    parser.add_argument('--epochs', type=int, default=25, help='Setting epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='The drop-out rate before each weight layer.')
    parser.add_argument('--optimizer', choices=('Adam', 'SGD', 'Adadelta'), help='Choose the optimizer.', default='Adam')
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate for each step.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Setting weight decay')
    parser.add_argument('--cap_uncertainty', action='store_true', help='Weather capture uncertainty.', default=False)
    parser.add_argument('--save_model_dir', type=str, default=os.path.join(base_dir, 'saved_models'), help='Set dir to save the model which has the best performance.')
    parser.add_argument('--data_csv', type=str, default=os.path.join(base_dir, 'data', 'DIAGNOSES_ICD.csv'), help='Path to data file')
    parser.add_argument('--icd9map', type=str, default=os.path.join(base_dir, 'data', 'icd9_map.json'), help='Path to json file for icd9 code mapping to categories')
    return parser.parse_args()


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
    return torch.mean(num / denom)


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
    return torch.mean(num / denom)


def eval_model(model, val_loader):
    y_pred = torch.FloatTensor()
    y_true = torch.FloatTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_hat = model(x, masks, rev_x, rev_masks)
        # y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    # ovr and ovo appear to give the same results for our model
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
    visit_lvl = visit_level_precision(5, y_true, y_pred)
    code_lvl = code_level_accuracy(5, y_true, y_pred)
    return roc_auc, visit_lvl, code_lvl


def train(model, train_loader, val_loader, n_epochs, params):
    train_start = time.time()

    optimizer = None
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if params['model'] in ['RNN', 'RNNplus', 'CNN', 'RETAIN', 'DIPOLE']:
        criterion = nn.BCELoss()
    elif params['model'] == 'INPREM':
        if params['cap_uncertainty']:
            criterion = UncertaintyLoss(params['task'], params['monto_carlo_for_aleatoric'],
                                        len(params['category2idx']))
        else:
            criterion = nn.BCELoss()
    else:
        raise Exception('unknown model type')

    torch.autograd.set_detect_anomaly(True)

    times = list()
    y_pred = torch.FloatTensor()
    y_true = torch.FloatTensor()
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, masks, rev_x, rev_masks)
            # print(y_hat)
            loss = criterion(y_hat, y.to(device))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        roc_auc, visit_lvl, code_lvl = eval_model(model, val_loader)
        print('Epoch: {} \t Validation roc_auc: {:.2f}, visit_lvl: {:.4f}, code_lvl: {:.4f}'.format(epoch + 1, roc_auc,
                                                                                                    visit_lvl,
                                                                                                    code_lvl))
        epoch_time = time.time() - epoch_start
        print('Epoch: {} \t Time elapsed: {:.2f} sec '.format(epoch + 1, epoch_time))
        times.append(epoch_time)
    print('Avg. time per epoch: {:.2f} sec '.format(sum(times) / len(times)))

    model_dir = os.path.join('./saved_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, f'{params["model"]}.pt'))
    return


def test(model, test_loader):
    y_pred = torch.FloatTensor()
    y_true = torch.FloatTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in test_loader:
        y_hat = model(x, masks, rev_x, rev_masks)
        # y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')

    visits = []
    codes = []
    for k in [5, 10, 15, 20, 25, 30]:
        visits.append((k, visit_level_precision(k, y_true, y_pred)))
        codes.append((k, code_level_accuracy(k, y_true, y_pred)))
    return roc_auc, visits, codes


if __name__ == '__main__':
    strt = time.time()
    opts = args()
    params = {
        'model': opts.model,
        'batch_size': opts.batch_size,
        'num_epochs': opts.epochs,
        'emb_dim': opts.emb_dim,
        'lr': opts.lr,
        'weight_decay': opts.weight_decay,
        'optimizer': opts.optimizer,
        'train': opts.train,
        'drop_rate': opts.drop_rate
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    load_start = time.time()
    dataset = CustomDataset(opts.data_csv, opts.icd9map)
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
    print('Time take to load data: {:.2f} sec '.format(time.time() - load_start))

    if params['model'] == 'RNN':
        model = RNN(len(dataset.idx2code), len(dataset.category2idx), params['emb_dim'], dataset.max_num_visits)
    elif params['model'] == 'RNNplus':
        model = RNNplus(len(dataset.idx2code), len(dataset.category2idx), params['emb_dim'])
    elif params['model'] == 'RETAIN':
        model = RETAIN(len(dataset.idx2code), len(dataset.category2idx), params['emb_dim'])
    elif params['model'] == 'DIPOLE':
        model = DIPOLE(len(dataset.idx2code), len(dataset.category2idx), params['emb_dim'], params['max_num_codes'])
    elif params['model'] == 'CNN':
        model = CNN(params)
    elif params['model'] == 'INPREM':
        params['task'] = 'diagnoses'
        params['n_depth'] = 2
        params['n_head'] = 2
        params['d_k'] = params['emb_dim']
        params['d_v'] = params['emb_dim']
        params['d_inner'] = params['emb_dim']
        params['cap_uncertainty'] = opts.cap_uncertainty
        params['dp'] = False
        params['dvp'] = False
        params['ds'] = False
        params['monto_carlo_for_epistemic'] = 200
        params['monto_carlo_for_aleatoric'] = 100

        max_visits = params['max_num_visits']
        input_dim = params['max_num_codes']
        output_dim = len(dataset.category2idx) if not params['cap_uncertainty'] else int(len(dataset.category2idx) / 2)
        model = Inprem(params['task'], input_dim, output_dim, params['emb_dim'], max_visits,
                       params['n_depth'], params['n_head'], params['d_k'], params['d_v'], params['d_inner'],
                       params['cap_uncertainty'], params['drop_rate'], False, params['dp'], params['dvp'],
                       params['ds'])
    else:
        raise Exception('unknown model type')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    if params['train']:
        train(model, train_loader, val_loader, params['num_epochs'], params)
    else:
        model_path = os.path.join('./saved_models', f'{params["model"]}.pt')
        if not os.path.exists(model_path):
            raise Exception(f'saved model does not exist: {model_path}')
        model.load_state_dict(torch.load(model_path))

    test_start = time.time()
    roc_auc, visit_prec, code_acc = test(model, test_loader)
    print('Test roc_auc: {:.4f}'.format(roc_auc))
    visit_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in visit_prec])
    print('Test visit-level precision@k: {}'.format(visit_str))
    code_str = ' '.join(['{:.4f}@{}'.format(v, k) for k, v in code_acc])
    print('Test code-level accuracy@k: {}'.format(code_str))
    print('Time take to test model: {:.2f} sec '.format(time.time() - test_start))
    print('Total time to run: {:.2f} sec '.format(time.time() - strt))
