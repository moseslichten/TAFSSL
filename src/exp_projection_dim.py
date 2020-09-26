# --------------------------------------------------------
# TAFSSL
# Copyright (c) 2019 IBM Corp
# Licensed under The Apache-2.0 License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
from utils.proto_msp import ProtoMSP
import time
import pickle
from utils.misc import print_params
from utils.misc import load_features
from utils.misc import print_msg
from utils.misc import avg, ci_95, parse_args
from utils.misc import create_episode, calc_acc
from utils.misc import get_color
from utils.misc import get_features


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15, args=None):
    model.train_mean = train_mean
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query)

    scores = []
    for d in args:
        model.opt.reduced_dim = d
        scores += [model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_ica), ]

    return calc_acc(scores, y)


def run_exp(params, verbose, args):
    print_params(params)
    n_episodes = 10000

    few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot)
    model = ProtoMSP(opt=params)
    model = model.cuda()
    train_mean, cl_data_file = load_features(params)

    acc_list = []
    start_time = time.perf_counter()
    for i in range(1, n_episodes + 1):
        acc = run_episode(train_mean, cl_data_file, model, n_query=params.n_query, **few_shot_params, args=args)
        acc_list += acc
        if i % verbose == 0:
            print_msg(i, n_episodes, start_time, acc_list, acc)
    res = [avg(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    ci = [ci_95(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    return res, ci


def projection_dim_exp_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for ds in ['mini', 'tiered']:
        ax = plt.subplot(111)
        with open(f'exp_projection_dim_{ds}.pickle', 'rb') as handle:
            exp_dict = pickle.load(handle)
            x = exp_dict['args']

        y_lim = np.inf
        for part in ['val', 'test']:
            name = f'{part}'
            res, _ = exp_dict[name]
            plt.plot(x, res, label=name, color=get_color(name), linewidth=2)
            y_lim = min(y_lim, np.min(res))

        ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=3)
        # plt.ylim(bottom=y_lim - 1)
        plt.grid()
        fig = plt.figure(1)
        plt.xlabel("reduced dimension")
        plt.ylabel("accuracy")
        plt.savefig(f"projection-dim-exp-{ds}.png")
        plt.close(fig)


def projection_dim_exp():
    args = [5, 10, 15, 20]
    params = parse_args('test')
    params.n_shot = 1
    for ds in ['mini', 'tiered']:
        params.dataset = ds
        exp_dict = {'args': args}
        for part in ['val', 'test']:
            params.part = part
            res, ci = run_exp(params, verbose=500, args=args)
            exp_dict[f'{part}'] = (res, ci)

        with open(f'exp_projection_dim_{ds}.pickle', 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    get_features()
    projection_dim_exp()
    projection_dim_exp_fig()
