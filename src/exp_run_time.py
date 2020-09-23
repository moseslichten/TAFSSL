import numpy as np
from utils.proto_msp import ProtoMSP
import time
import pickle
from utils.misc import print_params
from utils.misc import load_features
from utils.misc import parse_args
from utils.misc import create_episode
from utils.misc import get_features


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15):
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query)

    model.train_mean = train_mean

    time_list = [time.time(), ]

    model.opt.reduced_dim = 4
    model.method_sub(z_all, model.sub_train_mean)
    time_list.append(time.time())
    model.method_sub(z_all, model.mean_and_norm)
    time_list.append(time.time())
    model.method_sub(z_all, model.mean_and_norm_ber)
    time_list.append(time.time())
    model.method_project(z_all, model.mean_and_norm_ber, model.calc_pca)
    time_list.append(time.time())
    model.method_project(z_all, model.mean_and_norm_ber, model.calc_ica)
    time_list.append(time.time())

    model.opt.reduced_dim = 10
    model.method_cluster_baseline(z_all)
    time_list.append(time.time())
    model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_pca)
    time_list.append(time.time())
    model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_ica)
    time_list.append(time.time())

    model.method_mean_shift(z_all)
    time_list.append(time.time())

    model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_pca)
    time_list.append(time.time())
    model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_ica)
    time_list.append(time.time())

    time_list = [i - j for i, j in zip(time_list[1:], time_list[:-1])]
    return np.asarray(time_list)


def run_exp(params, verbose):
    print_params(params)
    n_episodes = 10000

    few_shot_params = dict(n_way=params.n_way, n_support=params.n_shot)
    model = ProtoMSP(opt=params)
    model = model.cuda()
    train_mean, cl_data_file = load_features(params)

    time_list = []
    name = ['baseline', 'sub reg', 'sub', 'pca', 'ica',
                'cluster', 'pca cluster', 'ica cluster',
                'msp', 'pca msp', 'ica msp',
                ]
    for i in range(1, n_episodes + 1):
        times = run_episode(train_mean, cl_data_file, model, n_query=params.n_query, **few_shot_params)
        time_list.append(times)
        if i % verbose == 0:
            tl = sum(time_list)/len(time_list)
            msg = ''
            for k, v in zip(name, tl):
                msg += f'{k}: {v:.2e}, '
            print(msg)
    return


def table_exp():
    params = parse_args('test')
    for ds in ['mini', 'tiered']:
        for shot in [1, 5]:
            params.dataset = ds
            params.n_shot = shot
            run_exp(params, verbose=500)


if __name__ == '__main__':
    get_features()
    table_exp()

