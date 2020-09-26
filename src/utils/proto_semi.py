# --------------------------------------------------------
# TAFSSL
# Copyright (c) 2019 IBM Corp
# Licensed under The Apache-2.0 License [see LICENSE for details]
# --------------------------------------------------------

from utils.proto_multi import ProtoMulti
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils.misc import pairwise_distances
from utils.misc import most_confident


def concat(z_support, z_semi, z_query, idx):
    """
    :param z_support: (n_way, n_supp, n_feat)
    :param z_semi: (n_way * n_semi, n_feat)
    :param z_query: (n_way * n_query, n_feat)
    :param idx: (n_way, k)
    :return:
    """
    pseudo = torch.stack([torch.index_select(z_semi, 0, idi) for idi in idx], dim=0)
    supp = torch.cat([z_support, pseudo], dim=1)
    supp = supp.view(-1, supp.shape[-1])
    return torch.cat([supp, z_semi, z_query], dim=0)


class ProtoSemi(ProtoMulti):
    def __init__(self, opt):
        super().__init__(opt)
        self.n_semi = opt.n_semi
        self.n_iter_msp = opt.n_iter_msp

    def parse_feature(self, x):
        z_support = x[:self.n_support * self.n_way]
        z_semi = x[self.n_support * self.n_way:self.n_support * self.n_way + self.n_semi * self.n_way]
        z_query = x[self.n_support * self.n_way + self.n_semi * self.n_way:]
        return z_support, z_semi, z_query

    def get_batch(self, x):
        z_support, z_semi, z_query = self.parse_feature(x)  # (n_way, n_support, n_feat), (n_way, n_query, n_feat)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_semi = z_semi.contiguous().view(self.n_way * self.n_semi, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        return z_support, z_semi, z_query

    def mean_and_norm(self, z_all):
        z_support, z_semi, z_query = self.parse_feature(z_all)
        x = torch.cat((z_support, z_semi), dim=0)
        z_all = z_all - x.mean(dim=0, keepdim=True)
        z_all = F.normalize(z_all, p=2, dim=-1)
        return z_all

    def mean_and_norm_ber(self, z_all):
        z_support, z_semi, z_query = self.parse_feature(z_all)
        z_support = z_support - z_support.mean(dim=0, keepdim=True)
        z_semi = z_semi - z_semi.mean(dim=0, keepdim=True)
        z_query = z_query - z_semi.mean(dim=0, keepdim=True)
        z_all = torch.cat((z_support, z_semi, z_query), dim=0)
        z_all = F.normalize(z_all, p=2, dim=-1)
        return z_all

    def cluster_semi(self, supp_and_semi, query):
        """
        z_all: (n_way * (n_support + n_query), n_feat)
        """
        z_sup = supp_and_semi[:self.n_way * self.n_support]
        supp_and_semi = supp_and_semi.cpu().numpy()
        kmeans = KMeans(n_clusters=self.k).fit(supp_and_semi)

        centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
        centers = F.normalize(centers, p=2, dim=-1)

        dist_supp = pairwise_distances(z_sup, centers)  # (n_way * n_supp, k)
        prob_support = torch.exp(-self.sigma * dist_supp)
        prob_support = prob_support / prob_support.sum(dim=-1, keepdim=True)

        prob_support = prob_support.view(self.n_way * self.n_support, self.k)
        prob_support = prob_support.transpose(-1, -2).view(self.k, -1, 1)

        dist_query = pairwise_distances(query, centers)  # (n_way * n_query, k)
        prob_query = torch.exp(-self.sigma * dist_query)
        prob_query = prob_query / prob_query.sum(dim=-1, keepdim=True)
        temperature = 0.1
        prob_query = F.softmax(prob_query/temperature, dim=-1)

        prob_query = prob_query.transpose(-1, -2).view(self.k, 1, -1)
        return prob_support, prob_query, centers

    def method_project_semi(self, z_all, mean_func, project_func):
        x = mean_func(z_all)
        z_support, z_semi, z_query = self.parse_feature(x)
        basis = project_func(torch.cat((z_support, z_semi), dim=0))
        x = self.project_and_norm(x, basis)
        z_support, z_semi, z_query = self.parse_feature(x)
        return self.calc_prob(z_support, z_query)

    def method_calc_prob(self, x):
        z_support, z_semi, z_query = self.parse_feature(x)
        return self.calc_prob(z_support, z_query), self.calc_prob(z_support, z_semi)

    def method_proj_and_cluster_semi(self, z_all, mean_func, project_func):
        x = mean_func(z_all)
        z_support, z_semi, z_query = self.parse_feature(x)
        basis = project_func(torch.cat((z_support, z_semi), dim=0))
        x = self.project_and_norm(x, basis)
        z_support, z_semi, z_query = self.parse_feature(x)

        prob_support, prob_query, centers = self.cluster_semi(torch.cat((z_support, z_semi), dim=0), z_query)

        x = torch.cat((z_support, z_query), dim=0)
        x = self.expand(x)
        prob_given_k = self.prob_given_cluster(x, prob_support)
        prob = self.bayesian_prob(prob_given_k, prob_query)
        return prob

    def method_mean_shift_semi(self, z_all, mean_func, project_func):
        n_support = self.n_support
        x = mean_func(z_all)
        z_support, z_semi, z_query = self.parse_feature(x)
        basis = project_func(torch.cat((z_support, z_semi), dim=0))
        z_all = self.project_and_norm(x, basis)

        z_support, z_semi, z_query = self.get_batch(z_all)
        z_support = z_support.view(self.n_way, self.n_support, -1)

        score_q, score_s = self.method_calc_prob(z_all)
        for _ in range(self.n_iter_msp):
            idx = most_confident(score_s)
            self.n_support = n_support + idx.shape[1]
            z_all = concat(z_support, z_semi, z_query, idx)
            score_q, score_s = self.method_calc_prob(z_all)

        self.n_support = n_support
        return score_q
