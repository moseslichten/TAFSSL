import numpy as np
from utils.proto_msp import ProtoMSP
import time
import pickle
from utils.misc import print_params
from utils.misc import load_features
from utils.misc import print_msg
from utils.misc import avg, ci_95, parse_args
from utils.misc import create_episode
from utils.misc import get_color
from utils.misc import get_features
import torch
import random


def run_exp(params, verbose):
    print_params(params)

    n_episodes = 10000

    episode_params = dict(n_way=params.n_way, n_support=params.n_shot,
                          n_query=params.n_query, n_unbalance_max=params.n_unbalance_max)

    model = ProtoMSP(opt=params)
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


def create_episode_unbalanced(cl_data_file, n_way=5, n_support=5, n_query=15, n_unbalance_max=15):
    if n_unbalance_max == 0:
        return

    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)  # List with the class idx

    n_unbalance = np.random.randint(n_unbalance_max, size=n_way)
    y_query = []
    z_support = []
    z_query = []
    for jj, cl in enumerate(select_class):
        img_feat = cl_data_file[cl]
        perm_ids = list(np.random.permutation(len(img_feat)).tolist())
        snq = np.array([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_support + n_query + n_unbalance[jj])])  # stack each batch
        z_support.append(snq[:n_support, :])
        z_query.append(snq[n_support:, :])
        y_query += [jj] * (n_query + n_unbalance[jj])
    z_support = np.array(z_support)  # (num ways, n_support + n_query, n_feat)
    z_query = np.array(z_query)  # (num ways, n_support + n_query, n_feat)
    z_support = z_support.reshape((n_way * n_support, -1))
    z_query = np.concatenate(z_query)  # .reshape((len(y_query), -1))
    y = np.asarray(y_query)

    perm = np.random.permutation(y.shape[0])
    z_query = np.take(z_query, perm, axis=0)
    y = np.take(y, perm, axis=0)

    z_all = np.concatenate((z_support, z_query))
    z_all = torch.from_numpy(np.array(z_all)).cuda()  # (num ways * (n_support + n_query) + n_distract * n_query, n_feat)
    return z_all, y


def calc_acc(scores, y):
    acc_list = []
    for each_score in scores:
        pred = each_score.data.cpu().numpy().argmax(axis=0)

        acc = np.sum(pred == y) / np.sum(y != -1)
        acc_list.append(acc * 100)
    return acc_list


def run_episode(train_mean, cl_data_file, model, n_way=5, n_support=5, n_query=15, n_unbalance_max=15):
    if n_unbalance_max > 0:
        z_all, y = create_episode_unbalanced(cl_data_file, n_way, n_support, n_query, n_unbalance_max)
    else:
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
    scores += [
        model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_pca),
        model.method_proj_and_cluster(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]

    scores += [
        model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_pca),
        model.method_project_and_mean_shift(z_all, model.mean_and_norm_ber, model.calc_ica),
               ]

    return calc_acc(scores, y)


def unbalanced_exp():
    params = parse_args('test')
    for ds in ['mini', 'tiered']:
        params.dataset = ds
        exp_dict = {}
        for q in [0, 10, 20, 30, 40, 50]:
            params.n_unbalance_max = q
            res, ci = run_exp(params, verbose=500)
            exp_dict[q] = (res, ci)
        with open(f'exp_unbalanced_{ds}.pickle', 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unbalanced_exp_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for ds in ['mini', 'tiered']:
        with open(f'exp_unbalanced_{ds}.pickle', 'rb') as handle:
            exp_dict = pickle.load(handle)

        x = np.array(list(exp_dict.keys()))
        values = np.array(list(exp_dict.values()))
        y = values[:, 0, :]

        ax = plt.subplot(111)
        exp_name = ['simple', 'sub', 'sub ber', 'pca', 'ica',
                    'pca & bkm', 'ica & bkm', 'pca & msp', 'ica & msp']
        y_lim = None
        for n in [*range(y.shape[1])]:
            if exp_name[n] in ['simple', 'sub', 'ica', 'ica & bkm', 'ica & msp']:
                plt.plot(x, y[:, n].reshape(-1, ), label=exp_name[n], color=get_color(exp_name[n]), linewidth=2)
                if exp_name[n] == 'simple':
                    y_lim = np.min(y[:, n])

        ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=3)

        plt.grid()
        fig = plt.figure(1)
        plt.xlabel("unbalanced factor")
        plt.ylabel("accuracy")
        plt.ylim(bottom=y_lim-3)
        # plt.xlim((np.min(x), np.max(x)))
        plt.savefig(f"exp-unbalanced-{ds}.png")
        plt.close(fig)


if __name__ == '__main__':
    get_features()
    unbalanced_exp()
    unbalanced_exp_fig()
