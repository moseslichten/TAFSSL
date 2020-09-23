import numpy as np
from utils.proto_semi import ProtoSemi
import time
import pickle
from utils.misc import print_params
from utils.misc import load_features
from utils.misc import print_msg
from utils.misc import avg, ci_95, parse_args
from utils.misc import calc_acc
from utils.misc import get_features
import torch
import random


def run_exp(params, verbose):
    print_params(params)

    n_episodes = 10000

    episode_params = dict(n_way=params.n_way, n_support=params.n_shot,
                          n_query=params.n_query, n_semi=params.n_semi)

    model = ProtoSemi(opt=params)
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


def create_episode(cl_data_file, n_way=5, n_support=5, n_query=15, n_semi=5):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)  # List with the class idx
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = list(np.random.permutation(len(img_feat)).tolist())
        z_all.append([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_support + n_semi + n_query)])  # stack each batch
    z_all = np.array(z_all)  # (num ways, n_support + n_query, n_feat)
    z_support = z_all[:, :n_support, :]
    z_support = z_support.reshape((n_way * n_support, -1))

    z_semi = z_all[:, n_support:n_support + n_semi, :]
    z_semi = z_semi.reshape((n_way * n_semi, -1))
    np.random.shuffle(z_semi)

    z_query = z_all[:, n_support + n_semi:, :]
    z_query = z_query.reshape((n_way * n_query, -1))
    y = np.repeat(range(n_way), n_query)

    perm = np.random.permutation(y.shape[0])
    z_query = np.take(z_query, perm, axis=0)
    y = np.take(y, perm, axis=0)

    z_all = np.concatenate((z_support, z_semi, z_query))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_all = torch.from_numpy(np.array(z_all)).to(device)  # (ways * (supp + semi + query), n_feat)
    return z_all, y


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15, n_semi=30):
    z_all, y = create_episode(cl_data_file, n_way, n_support, n_query, n_semi)
    model.train_mean = train_mean

    scores = []
    model.opt.reduced_dim = 4
    scores += [model.method_project_semi(z_all, model.mean_and_norm, model.calc_ica), ]

    model.opt.reduced_dim = 10
    scores += [model.method_proj_and_cluster_semi(z_all, model.mean_and_norm, model.calc_ica), ]
    scores += [model.method_mean_shift_semi(z_all, model.mean_and_norm, model.calc_ica), ]

    return calc_acc(scores, y)


def semi_exp():
    params = parse_args('test')
    exp_dict = {}
    for ds in ['mini', 'tiered']:
        for shot in [1, 5]:
            for num_semi in [30, 50, 100]:
                params.dataset = ds
                params.n_shot = shot
                params.n_semi = num_semi
                res, ci = run_exp(params, verbose=500)
                exp_dict[f'{ds} {shot} shot {num_semi} unlabeled'] = (res, ci)

    with open('exp_semi.pickle', 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def semi_exp_res():
    with open('exp_semi.pickle', 'rb') as handle:
        exp_dict = pickle.load(handle)
    for k, v in exp_dict.items():
        print(f'{k}: {np.around(v, 2)}')


def table_latex():
    with open('exp_semi.pickle', 'rb') as handle:
        exp_dict = pickle.load(handle)

    pm = "$\pm$"
    name = ['ica', 'ica cluster', 'ica msp', ]
    for i in range(len(name)):
        for num_semi in [30, 50, 100]:
            line = f'{name[i]} {num_semi}\n'
            for ds in ['mini', 'tiered']:
                for shot in [1, 5]:
                    acc, ci = exp_dict[f'{ds} {shot} shot {num_semi} unlabeled']
                    line += f'{acc[i]:.2f} {pm} {ci[i]:.2f}'
                    if ds == 'tiered' and shot == 5:
                        line += "\\\\"
                    else:
                        line += ' & '
            print(line)


if __name__ == '__main__':
    get_features()
    semi_exp()
    semi_exp_res()
    table_latex()
