import numpy as np
import argparse
import pickle
import time
import torch
import random
import requests


def load_features(params):
    features_path = f'{params.feats_path}/{params.dataset}_{params.arch}_{params.mode}_{params.part}.pickle'
    with open(features_path, 'rb') as handle:
        train_mean, cl_data_file = pickle.load(handle)
    return train_mean, cl_data_file


def avg(l):
    return np.mean(np.array(l))


def ci_95(l):
    return 1.96 * np.std(np.array(l)) / np.sqrt(len(l))


def parse_args(script):
    parser = argparse.ArgumentParser(description=f'few-shot script {script}')
    parser.add_argument('--feats-path', default='features', help='path to saved features')
    parser.add_argument('--dataset', default='mini', help='mini, tiered')
    parser.add_argument('--arch', default='densenet', help='densenet, wrn, ')
    parser.add_argument('--mode', default='best', help='best, last')
    parser.add_argument('--part', default='test', help='test, val')

    parser.add_argument('--n_way', default=5, type=int, help='class num to classify')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int, help='number of queries')

    parser.add_argument('--reduced_dim', default=4, type=int, help='dim')
    parser.add_argument('--n_clusters', default=5, type=int, help='number of clusters')
    parser.add_argument('--n_iter_msp', default=4, type=int, help='number of iterations')
    parser.add_argument('--sigma', default=1, type=int, help='sigma')
    return parser.parse_args()


def seconds_to_str(sec):
    ep_m, ep_s = divmod(sec, 60)
    ep_h, ep_m = divmod(ep_m, 60)
    sec_str = "%d:%02d:%02d" % (ep_h, ep_m, ep_s)
    return sec_str


def time_stats(tot_iter, curr, seconds):
    time_per_iter = seconds / curr
    remained_iter = tot_iter - curr
    remained_str = seconds_to_str(time_per_iter * remained_iter)
    passed_str = seconds_to_str(seconds)
    tot_str = seconds_to_str(time_per_iter * tot_iter)
    return remained_str, passed_str, tot_str


def print_params(params):
    opt_msg = ''
    for key, value in vars(params).items():
        opt_msg += f'{key}: {value}\n'
    print(opt_msg)
    return


def print_msg(i, n_episodes, start_time, acc_list, acc):
    time_rm, time_pass, time_tot = time_stats(n_episodes, i, time.perf_counter() - start_time)
    msg = f'Episode [{i}/{n_episodes}] {time_rm} {time_pass} {time_tot} '
    for ind in range(len(acc)):
        msg += f'{avg(acc_list[ind::len(acc)]):.2f} ({ci_95(acc_list[ind::len(acc)]):.2f}) '
    print(msg)


def create_episode(cl_data_file, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)  # List with the class idx
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = list(np.random.permutation(len(img_feat)).tolist())
        z_all.append([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_support + n_query)])  # stack each batch
    z_all = np.array(z_all)  # (num ways, n_support + n_query, n_feat)
    z_support = z_all[:, :n_support, :]
    z_support = z_support.reshape((n_way * n_support, -1))
    z_query = z_all[:, n_support:, :]
    z_query = z_query.reshape((n_way * n_query, -1))
    y = np.repeat(range(n_way), n_query)

    perm = np.random.permutation(y.shape[0])
    z_query = np.take(z_query, perm, axis=0)
    y = np.take(y, perm, axis=0)

    z_all = np.concatenate((z_support, z_query))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_all = torch.from_numpy(np.array(z_all)).to(device)  # (num ways * (n_support + n_query), n_feat)
    return z_all, y


def calc_acc(scores, y):
    """
    :param scores: (n_way, n_queries)
    :param y: (n_queries, )
    :return:
    """
    acc_list = []
    for each_score in scores:
        pred = each_score.data.cpu().numpy().argmax(axis=0)
        acc_list.append(np.mean(pred == y) * 100)
    return acc_list


def pairwise_distances(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    assert x.shape[-1] == y.shape[-1]
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist


def pairwise_distances_batch(x, y):
    """
    Input: x dim (B, M, Q).
    Input: y dim (B, N, Q).
    Output: dist is a (C, M, N) , C distance matrices.
    """
    b = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]

    x_norm = (x ** 2).sum(-1).view(b, m, 1)
    y_t = torch.transpose(y, -1, -2)
    y_norm = (y ** 2).sum(-1).view(b, 1, n)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist


def most_confident(scores, thresh=0.3):
    """
    :param scores: (n_way, n_queries)
    :param thresh: float
    :return:
    """
    confidence, idx = torch.sort(scores, dim=1, descending=True)
    d = (confidence.min(dim=0, keepdim=True).values > thresh).sum()
    idx = idx[:, :d]
    return idx


def get_color(method):
    assert method in ['simple', 'sub', 'ica', 'ica & bkm', 'ica & msp'] + ['val', 'test'] + ['lst', 'tpn', 'skm']
    col = None
    if method in ['simple', 'skm']:
        col = [0.4, 0.8, 0.5]
    elif method in ['sub', 'tpn']:
        col = [0.2, 0.8, 1.0]
    elif method in ['lst']:
        col = [0.3, 0.7, 0.8]
    elif method in ['ica', 'val']:
        col = [0.1, 0.5, 1.0]
    elif method in ['ica & bkm', 'test']:
        col = [1.0, 0.0, 0.3]
    elif method == 'ica & msp':
        col = [1.0, 0.6, 0.3]
    return col


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_features():
    file_ids = ['xxx',
                'xxx',
                'xxx',
                'xxx']

    file_names = ['mini_densenet_best_test.pickle',
                  'mini_densenet_best_val.pickle',
                  'tiered_densenet_best_test.pickle',
                  'tiered_densenet_best_val.pickle']

    assert False, '\n\n***The link was deleted due to the ECCV policy of not attaching links, ' \
                  'it will be reinstated upon acceptance of the paper***\n\n'
    from pathlib import Path
    import os
    Path("features").mkdir(parents=True, exist_ok=True)

    for file_id, file_name in zip(file_ids, file_names):
        file_path = os.path.join("features", file_name)
        if not Path(file_path).is_file():
            download_file_from_google_drive(file_id, file_path)
    return
