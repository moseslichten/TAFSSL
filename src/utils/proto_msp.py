# --------------------------------------------------------
# TAFSSL
# Copyright (c) 2019 IBM Corp
# Licensed under The Apache-2.0 License [see LICENSE for details]
# --------------------------------------------------------

from utils.proto_multi import ProtoMulti
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import most_confident


def concat(z_support, z_query, idx):
    """
    :param z_support: (n_way, n_supp, n_feat)
    :param z_query: (n_way * n_query, n_feat)
    :param idx: (n_way, k)
    :return:
    """
    pseudo = torch.stack([torch.index_select(z_query, 0, idi) for idi in idx], dim=0)
    supp = torch.cat([z_support, pseudo], dim=1)
    supp = supp.view(-1, supp.shape[-1])
    return torch.cat([supp, z_query], dim=0)


class ProtoMSP(ProtoMulti):
    def __init__(self, opt):
        super().__init__(opt)
        self.n_iter_msp = opt.n_iter_msp

    def method_calc_prob(self, z_all):
        x = F.normalize(z_all, p=2, dim=-1)  # (n_way, n_support + n_query, n_feat)
        z_support, z_query = self.get_batch(x)  # (n_way * n_support, n_feat)
        return self.calc_prob(z_support, z_query)

    def method_mean_shift(self, z_all):
        z_all = self.sub_train_mean(z_all)

        n_support = self.n_support
        z_support, z_query = self.get_batch(z_all)
        z_support = z_support.view(self.n_way, self.n_support, -1)

        scores = self.method_calc_prob(z_all)
        for _ in range(self.n_iter_msp):
            idx = most_confident(scores)
            self.n_support = n_support + idx.shape[1]
            z_all = concat(z_support, z_query, idx)
            scores = self.method_calc_prob(z_all)
            self.n_support = n_support
        return scores

    def method_project_and_mean_shift(self, z_all, mean_func, project_func):
        x = mean_func(z_all)
        basis = project_func(x)
        z_all = self.project_and_norm(x, basis)

        n_support = self.n_support
        z_support, z_query = self.get_batch(z_all)
        z_support = z_support.view(self.n_way, self.n_support, -1)

        scores = self.method_calc_prob(z_all)
        for _ in range(self.n_iter_msp):
            idx = most_confident(scores)
            self.n_support = n_support + idx.shape[1]
            z_all = concat(z_support, z_query, idx)
            scores = self.method_calc_prob(z_all)
            self.n_support = n_support
        return scores
