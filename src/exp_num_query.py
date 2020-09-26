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


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15):
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query)

    model.train_mean = train_mean

    model.opt.reduced_dim = 4
    scores = [model.method_sub(z_all, model.sub_train_mean),
              model.method_sub(z_all, model.mean_and_norm_ber),
              model.method_project(z_all, model.mean_and_norm_ber, model.calc_ica),
              ]

    model.opt.reduced_dim = 10
    scores += [
        model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]

    scores += [
        model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]

    return calc_acc(scores, y)


def run_exp(params, verbose):
    print_params(params)
    n_episodes = 10000

    few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot)
    model = ProtoMSP(opt=params)
    model = model.cuda()
    train_mean, cl_data_file = load_features(params)

    acc_list = []
    start_time = time.perf_counter()
    for i in range(1, n_episodes + 1):
        acc = run_episode(train_mean, cl_data_file, model, n_query=params.n_query, **few_shot_params)
        acc_list += acc
        if i % verbose == 0:
            print_msg(i, n_episodes, start_time, acc_list, acc)
    res = [avg(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    ci = [ci_95(acc_list[ind::len(acc)]) for ind in range(len(acc))]
    return res, ci


def n_query_exp():
    params = parse_args('test')
    for ds in ['mini', 'tiered']:
        params.dataset = ds
        exp_dict = {}
        for q in [2, 5, 10, 15, 30, 40, 50]:
            params.n_query = q
            res, ci = run_exp(params, verbose=1000)
            exp_dict[q] = (res, ci)
        with open(f'exp_n_queries_{ds}.pickle', 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def n_query_exp_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for ds in ['mini', 'tiered']:
        with open(f'exp_n_queries_{ds}.pickle', 'rb') as handle:
            exp_dict = pickle.load(handle)

        x = np.array(list(exp_dict.keys()))
        values = np.array(list(exp_dict.values()))
        y = values[:, 0, :]

        ax = plt.subplot(111)
        exp_name = ['simple', 'sub', 'ica', 'ica & bkm', 'ica & msp']
        y_lim = None
        for n in [*range(y.shape[1])]:
            # plt.plot(x, y[:, n].reshape(-1, ), label=exp_name[n])
            plt.plot(x[1:], y[:, n].reshape(-1, )[1:], label=exp_name[n], color=get_color(exp_name[n]), linewidth=2)
            if exp_name[n] == 'simple':
                y_lim = np.min(y[:, n])

        ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=3)
        plt.ylim(bottom=y_lim - 3)
        plt.grid()
        fig = plt.figure(1)
        plt.xlabel("queries")
        plt.ylabel("accuracy")
        plt.savefig(f"num-query-exp-{ds}.png")
        plt.close(fig)


if __name__ == '__main__':
    get_features()
    n_query_exp()
    n_query_exp_fig()
