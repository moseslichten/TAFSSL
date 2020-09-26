# --------------------------------------------------------
# TAFSSL
# Copyright (c) 2019 IBM Corp
# Licensed under The Apache-2.0 License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
from utils.proto_semi import ProtoSemi
import time
import pickle
from utils.misc import print_params
from utils.misc import load_features
from utils.misc import print_msg
from utils.misc import avg, ci_95, parse_args
from utils.misc import get_color
from utils.misc import calc_acc
from utils.misc import get_features
import torch
import random


class ProtoSemiNoise(ProtoSemi):
    def __init__(self, opt):
        super().__init__(opt)
        self.n_distract = None

    def parse_feature(self, x):
        z_support = x[:self.n_support * self.n_way]
        z_semi = x[self.n_support * self.n_way:self.n_support * self.n_way + self.n_semi * (self.n_way + self.n_distract)]
        z_query = x[self.n_support * self.n_way + self.n_semi * (self.n_way + self.n_distract):]
        return z_support, z_semi, z_query

    def get_batch(self, x):
        z_support, z_semi, z_query = self.parse_feature(x)  # (n_way, n_support, n_feat), (n_way, n_query, n_feat)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_semi = z_semi.contiguous().view((self.n_way + self.n_distract) * self.n_semi, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        return z_support, z_semi, z_query


def run_exp(params, verbose):
    print_params(params)

    n_episodes = 10000
    episode_params = dict(n_way=params.n_way, n_support=params.n_shot,
                          n_query=params.n_query, n_semi=params.n_semi, n_distract=params.n_distract)

    model = ProtoSemiNoise(opt=params)
    model = model.cuda()

    train_mean, cl_data_file = load_features(params)

    acc_list = []
    start_time = time.perf_counter()
    for i in range(1, n_episodes + 1):
        acc = run_episode(train_mean, cl_data_file, model, **episode_params)
        acc_list += acc
        if i % verbose == 0:
            print_msg(i, n_episodes, start_time, acc_list, acc)
    res = [avg(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    ci = [ci_95(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    return res, ci


def create_episode(cl_data_file, n_way=5, n_support=5, n_query=15, n_semi=5, n_distract=5):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way + n_distract)  # List with the class idx
    z_all = []
    for cl in select_class[:n_way]:
        img_feat = cl_data_file[cl]
        perm_ids = list(np.random.permutation(len(img_feat)).tolist())
        z_all.append([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_support + n_semi + n_query)])  # stack each batch
    z_all = np.array(z_all)  # (num ways, n_support + n_query, n_feat)

    z_all = np.array(z_all)  # (num ways, n_support + n_query, n_feat)
    z_support = z_all[:, :n_support, :]
    z_support = z_support.reshape((n_way * n_support, -1))

    z_semi = z_all[:, n_support:n_support + n_semi, :]
    z_semi = z_semi.reshape((n_way * n_semi, -1))
    np.random.shuffle(z_semi)

    z_query = z_all[:, n_support + n_semi:, :]
    z_query = z_query.reshape((n_way * n_query, -1))
    y = np.repeat(range(n_way), n_query)

    if n_distract > 0:
        z_noise = []
        for cl in select_class[n_way:]:
            img_feat = cl_data_file[cl]
            perm_ids = list(np.random.permutation(len(img_feat)).tolist())
            z_noise.append([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_semi)])  # stack each batch
        z_noise = np.array(z_noise)  # (n_distract, n_query, n_feat)
        z_noise = z_noise.reshape((n_distract * n_semi, -1))
        z_semi = np.concatenate((z_semi, z_noise))
        np.random.shuffle(z_semi)

    perm = np.random.permutation(y.shape[0])
    z_query = np.take(z_query, perm, axis=0)
    y = np.take(y, perm, axis=0)

    z_all = np.concatenate((z_support, z_semi, z_query))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_all = torch.from_numpy(np.array(z_all)).to(device)  # (num ways * (n_support + n_query) + n_distract * n_query, n_feat)
    return z_all, y


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15, n_semi=5, n_distract=5):
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query, n_semi, n_distract)

    model.train_mean = train_mean
    model.n_distract = n_distract

    scores = []
    model.opt.reduced_dim = 4
    scores += [model.method_project_semi(z_all, model.mean_and_norm, model.calc_ica), ]

    model.opt.reduced_dim = 10
    scores += [model.method_proj_and_cluster_semi(z_all, model.mean_and_norm, model.calc_ica), ]
    scores += [model.method_mean_shift_semi(z_all, model.mean_and_norm, model.calc_ica), ]

    return calc_acc(scores, y)


def noise_exp():
    params = parse_args('test')
    params.n_semi = 100
    for ds in ['mini', 'tiered']:
        params.dataset = ds
        exp_dict = {}
        for q in range(0, 7+1):
            params.n_distract = q
            res, ci = run_exp(params, verbose=500)
            exp_dict[q] = (res, ci)
        with open(f'exp_noise_semi_{ds}.pickle', 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def noise_exp_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for ds in ['mini', 'tiered']:
        with open(f'exp_noise_semi_{ds}.pickle', 'rb') as handle:
            exp_dict = pickle.load(handle)

        x = np.array(list(exp_dict.keys()))
        values = np.array(list(exp_dict.values()))
        y = values[:, 0, :]

        x_plot = range(x.shape[0])

        ax = plt.subplot(111)
        exp_name = ['ica', 'ica & bkm', 'ica & msp']
        y_lim = None
        for n in [*range(y.shape[1])]:
            plt.plot(x_plot, y[:, n].reshape(-1, ), label=exp_name[n], color=get_color(exp_name[n]), linewidth=2)
            if exp_name[n] == 'ica & msp':
                y_lim = np.max(y[:, n])

        if ds == 'mini':
            compare_works = {
                'lst': [70.1, 68.0, 66.00, 64.0, 62.4, 60.8, 60.4, 60.0],
                'tpn': [62.7, 62.4, 61.85, 61.3, 61.00, 60.7, 59.95, 59.2],
                'skm': [62.1, 61.9, 61.40, 60.9, 60.45, 60.0, 59.30, 58.6],
            }
        else:
            compare_works = {
                'lst': [77.7, 76.25, 74.88, 73.5, 72.10, 70.7, 69.35, 68.0],
                'tpn': [72.1, 72.00, 71.75, 71.5, 71.20, 70.9, 69.65, 68.4],
                'skm': [68.6, 67.50, 67.25, 67.0, 66.10, 65.2, 64.88, 64.0],
            }
        for k, v in compare_works.items():
            plt.plot(x_plot, v, label=k, color=get_color(k), linewidth=2)

        ax.legend(loc='upper center', fancybox=True, shadow=True, ncol=3)
        plt.ylim(top=y_lim + 3)
        plt.xticks(x_plot, x)
        plt.grid()
        fig = plt.figure(1)
        plt.xlabel("distracting classes")
        plt.ylabel("accuracy")
        plt.savefig(f"exp-noise-semi-{ds}.png")
        plt.close(fig)


if __name__ == '__main__':
    get_features()
    noise_exp()
    noise_exp_fig()
