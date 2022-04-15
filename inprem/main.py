#  -*- coding: utf-8 -*-
# @Time : 2019/11/14 下午5:19
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn
import os
import argparse
import torch
import torch.nn.functional as F
from doctor.model import Inprem
from Loss import UncertaintyLoss


# diagnoses bestsetting batch 32 lr 0.0005 l2 0.0001 drop 0.5 emb 256 starval 50 end val 65
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('diagnoses', 'heart', 'diabetes', 'kidney'),
                        help='Choose a task.', default='heart')
    parser.add_argument('--data_root', type=str, default='../../datasets/',
                        help='The dataset root dir.')
    parser.add_argument('--fold', choices=(1, 2, 3, 4, 5), default=1, help='Choose a fold.')

    parser.add_argument('--use_cuda', action='store_true',
                        help='If use GPU.', default=True)
    parser.add_argument('--gpu_devices', type=str, default='0',
                        help='Choose devices ID for GPU.')
    parser.add_argument('--epochs', type=int, default=25, help='Setting epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini-batch size')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='The drop-out rate before each weight layer.')
    parser.add_argument('--optimizer', choices=('Adam', 'SGD', 'Adadelta'),
                        help='Choose the optimizer.', required=False)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='The learning rate for each step.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Setting weight decay')

    parser.add_argument('--n_head', type=int, default=2,
                        help='The number of head of self-attention for the visit attention.')
    parser.add_argument('--n_depth', type=int, default=2,
                        help='The number of layers of self-attention for the visit attention.')

    parser.add_argument('--emb_dim', type=int, default=128,
                        help='The size of medical variable (or code) embedding.')

    parser.add_argument('--d_k', type=int, default=128,
                        help='The size of vector before self attention ')
    parser.add_argument('--d_v', type=int, default=128,
                        help='The size of vector before self attention ')
    parser.add_argument('--d_inner', type=int, default=128,
                        help='')
    parser.add_argument('--dvp', action='store_true', default=False,
                        help='Weather use position embedding.')
    parser.add_argument('--dp', action='store_true', default=False,
                        help='Weather use position embedding.')
    parser.add_argument('--ds', action='store_true', default=False, help='whether delete the sparse_max')
    parser.add_argument('--cap_uncertainty', action='store_true',
                        help='Weather capture uncertainty.', default=True)
    parser.add_argument('--monto_carlo_for_aleatoric', type=int, default=100,
                        help='The size of Monto Carlo Sample.')
    parser.add_argument('--monto_carlo_for_epistemic', type=int, default=200,
                        help='The size of Monto Carlo Sample.')
    parser.add_argument('--analysis_dir', type=str, default='../../output_for_analysis_final/',
                        help='Set the analysis output dir')
    parser.add_argument('--write_performance', action='store_true', default=False,
                        help='Weather write performance result')
    parser.add_argument('--performance_dir', type=str, default='../../metric_results/',
                        help='Set the performance dir')
    parser.add_argument('--save_model_dir', type=str, default='../../saved_model',
                        help='Set dir to save the model which has the best performance.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Choose the model dict to load for test or fine-tune.')
    parser.add_argument('--data_scale', default=1, type=float)
    return parser


def monto_calo_test(net, seq, mask, T):
    out, aleatoric = None, None
    outputs = []
    for i in range(T):
        seed = random.randint(0, 100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        out_instance = net(seq, mask)
        aleatoric = out_instance[:, 2:]
        if i == 0:
            out = F.softmax(out_instance[:, :2], dim=1)
            outputs.append(F.softmax(out_instance[:, :2], dim=1).cpu().detach().numpy())
        else:
            out = out + F.softmax(out_instance[:, :2], dim=1)
            aleatoric = aleatoric + out_instance[:, 2:]
            outputs.append(F.softmax(out_instance[:, :2], dim=1).cpu().detach().numpy())
    out = out / T
    aleatoric = aleatoric / T
    epistemic = -torch.sum(out * torch.log(out), dim=1)

    return out, aleatoric, epistemic, outputs

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_devices
    # TODO: Load dataset

    '''Define the model.'''
    net = Inprem(opts.task, input_dim, 2, opts.emb_dim,
                 max(train_set.max_visit, valid_set.max_visit, test_set.max_visit),
                 opts.n_depth, opts.n_head, opts.d_k, opts.d_v, opts.d_inner,
                 opts.cap_uncertainty, opts.drop_rate, False, opts.dp, opts.dvp, opts.ds)

    '''Select loss function'''
    if opts.cap_uncertainty:
        criterion = UncertaintyLoss(opts.task, opts.monto_carlo_for_aleatoric, 2)
    else:
        criterion = CrossEntropy(opts.task)

    if opts.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    '''Select optimizer'''
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)


    #TODO: Training, validating, and testing, In valid and test phase, you should use the monto_calo_test().

    # monto_calo_test(net, input, mask, opts.monto_carlo_for_epistemic)


if __name__ == '__main__':
    opts = args().parse_args()
    main(opts)