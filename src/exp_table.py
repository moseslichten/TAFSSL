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
from utils.misc import get_features


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15):
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query)

    model.train_mean = train_mean

    model.opt.reduced_dim = 4
    scores = [model.method_sub(z_all, model.sub_train_mean),
              model.method_sub(z_all, model.mean_and_norm),
              model.method_sub(z_all, model.mean_and_norm_ber),
              model.method_project(z_all, model.mean_and_norm_ber, model.calc_pca),
              model.method_project(z_all, model.mean_and_norm_ber, model.calc_ica),
              ]

    model.opt.reduced_dim = 10
    scores += [model.method_cluster_baseline(z_all),
               model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_pca),
               model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]

    scores += [model.method_mean_shift(z_all),
               model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_pca),
               model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]
    return calc_acc(scores, y)


def run_exp(params, verbose):
    print_params(params)
    n_episodes = 10000

    few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot)
    model = ProtoMSP(opt=params)
    model = model.to(model.device)
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


def table_exp():
    params = parse_args('test')
    exp_dict = {}
    for ds in ['mini', 'tiered']:
        for shot in [1, 5]:
            params.dataset = ds
            params.n_shot = shot
            res, ci = run_exp(params, verbose=500)
            exp_dict[f'{ds} {shot} shot'] = (res, ci)

    with open('exp_table.pickle', 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def table_exp_res():
    with open('exp_table.pickle', 'rb') as handle:
        exp_dict = pickle.load(handle)
    for k, v in exp_dict.items():
        print(f'{k}: {np.around(v, 2)}')


def table_latex():
    with open('exp_table.pickle', 'rb') as handle:
        exp_dict = pickle.load(handle)

    pm = "$\pm$"
    name = ['baseline', 'sub reg', 'sub', 'pca', 'ica',
            'cluster', 'pca cluster', 'ica cluster',
            'msp', 'pca msp', 'ica msp',
            ]
    for i in range(len(name)):
        line = f'{name[i]}\n'
        for ds in ['mini', 'tiered']:
            for shot in [1, 5]:
                acc, ci = exp_dict[f'{ds} {shot} shot']
                line += f'{acc[i]:.2f} {pm} {ci[i]:.2f}'
                if ds == 'tiered' and shot == 5:
                    line += "\\\\"
                else:
                    line += ' & '

        print(line)


if __name__ == '__main__':
    get_features()
    table_exp()
    table_exp_res()
    table_latex()
