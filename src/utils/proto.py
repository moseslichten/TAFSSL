import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from utils.misc import pairwise_distances


class ProtoNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_way = opt.n_way
        self.n_support = opt.n_shot
        self.n_query = opt.n_query
        self.sigma = opt.sigma
        self.train_mean = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        return x

    def parse_feature(self, x):
        z_support = x[:(self.n_support * self.n_way)]
        z_query = x[(self.n_support * self.n_way):]
        return z_support, z_query

    def get_batch(self, x):
        z_support, z_query = self.parse_feature(x)  # (n_way, n_support, n_feat), (n_way, n_query, n_feat)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(z_query.shape[0], -1)
        return z_support, z_query

    def mean_and_norm(self, z_all):
        z_all = z_all - torch.mean(z_all, dim=0, keepdim=True)
        z_all = F.normalize(z_all, p=2, dim=-1)  # (n_way, n_support + n_query, n_feat)
        return z_all

    def mean_and_norm_ber(self, z_all):
        z_support, z_query = self.parse_feature(z_all)
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_support = z_support - z_support.mean(dim=0, keepdim=True)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)

        z_query = z_query - z_query.mean(dim=0, keepdim=True)
        z_all = torch.cat((z_support, z_query), dim=0)
        z_all = F.normalize(z_all, p=2, dim=-1)  # (-1, n_feat)
        return z_all

    def sub_train_mean(self, z_all):
        x = z_all - torch.from_numpy(self.train_mean).to(self.device)
        x = F.normalize(x, p=2, dim=-1)  # (n_way, n_support + n_query, n_feat)
        return x

    def calc_prob(self, z_support, z_query):
        z_support_mean = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_support_mean = z_support_mean.mean(dim=-2)  # (n_way , n_feat)
        pair_dist = pairwise_distances(z_support_mean, z_query)
        prob = torch.exp(-self.sigma * pair_dist)
        prob = prob / prob.sum(dim=0, keepdim=True)
        return prob

    def calc_pca(self, x):
        _, _, v = torch.svd(x.view(-1, x.shape[-1]).cpu())  # (n_feat, n_ex)
        v = v[:, :self.opt.reduced_dim].to(self.device)  # (n_feat, dim)
        return v

    def calc_ica(self, x):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('error')
        x_np = x.view(-1, x.shape[-1]).cpu().numpy().transpose()
        s = None
        while s is None:
            try:
                s = FastICA(n_components=self.opt.reduced_dim).fit_transform(x_np).astype(np.float32)
            except (RuntimeWarning, ConvergenceWarning) as e:
                print(f'{e}')
                pass
        ica = torch.from_numpy(s).to(self.device)
        return ica

    def calc_fa(self, x):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('error')
        x_np = x.view(-1, x.shape[-1]).cpu().numpy().transpose()
        s = None
        while s is None:
            try:
                s = FactorAnalysis(n_components=self.opt.reduced_dim).fit_transform(x_np).astype(np.float32)
            except (RuntimeWarning, ConvergenceWarning) as e:
                print(f'{e}')
                pass
        fa = torch.from_numpy(s).to(self.device)
        return fa

    @staticmethod
    def project_and_norm(x, base):
        return F.normalize(x @ base, p=2, dim=-1)

    def method_wo_sub(self, z_all):
        x = F.normalize(z_all, p=2, dim=-1)  # (n_way, n_support + n_query, n_feat)
        z_support, z_query = self.get_batch(x)  # (n_way * n_support, n_feat)
        return self.calc_prob(z_support, z_query)

    def method_sub(self, z_all, mean_func):
        x = mean_func(z_all)
        z_support, z_query = self.get_batch(x)  # (n_way * n_support, n_feat)
        return self.calc_prob(z_support, z_query)

    def method_project(self, z_all, mean_func, project_func):
        x = mean_func(z_all)
        basis = project_func(x)
        x = self.project_and_norm(x, basis)
        z_support, z_query = self.get_batch(x)  # (n_way * n_support, n_feat)
        return self.calc_prob(z_support, z_query)
