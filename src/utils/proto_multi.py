from utils.proto import ProtoNet
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import pairwise_distances
from utils.misc import pairwise_distances_batch


class ProtoMulti(ProtoNet):
    def __init__(self, opt):
        super().__init__(opt)
        self.k = opt.n_clusters

    def cluster(self, z_all):
        """
        z_all: (n_way * (n_support + n_query), n_feat)
        """
        z_sup = z_all[:self.n_way * self.n_support]
        z_all = z_all.cpu().numpy()
        kmeans = KMeans(n_clusters=self.k).fit(z_all)

        prob = kmeans.labels_
        prob = np.eye(self.k)[prob].astype(np.float32)

        centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
        centers = F.normalize(centers, p=2, dim=-1)

        dist = pairwise_distances(z_sup, centers)  # (n_way * n_supp, k)
        prob_support = torch.exp(-self.sigma * dist)
        prob_support = prob_support / prob_support.sum(dim=-1, keepdim=True)

        prob_support = prob_support.view(self.n_way * self.n_support, self.k)
        prob_support = prob_support.transpose(-1, -2).view(self.k, -1, 1)

        prob_query = torch.from_numpy(prob[(self.n_way * self.n_support):]).to(self.device)
        prob_query = prob_query.view(-1, self.k)
        prob_query = prob_query.transpose(-1, -2).view(self.k, 1, -1)

        centers = centers.view(self.k, 1, -1)
        return prob_support, prob_query, centers

    def prob_given_cluster(self, x, prob_support):
        """
        :param x: (k, n_way * n_sup, n_way * n_query)
        :param prob_support: (k, n_way * n_support, 1)
        :return: (k, n_way, n_way * n_query)
        """
        z_support = x[:, :(self.n_way * self.n_support)]
        z_query = x[:, (self.n_way * self.n_support):]
        pair_dist = pairwise_distances_batch(z_support, z_query)  # (k, n_way * n_sup, n_way * n_query)
        pair_dist = torch.exp(-self.sigma * pair_dist)  # (k, n_way * n_sup, n_way * n_query)
        prob_given_k = pair_dist * prob_support  # (k, n_way * n_sup, n_way * n_query)
        prob_given_k = prob_given_k / prob_given_k.sum(dim=1, keepdim=True)
        prob_given_k = prob_given_k.view(self.k, self.n_way, self.n_support, -1).sum(dim=2)  # (k, n_way, n_way * n_query)
        return prob_given_k

    @staticmethod
    def bayesian_prob(prob_given_k, prob_query):
        """
        :param prob_given_k: (k, n_way, n_way * n_query)
        :param prob_query: (k, 1, n_way * n_query)
        :return: (n_way, n_way * n_query)
        """
        prob = prob_given_k * prob_query
        prob = prob.sum(dim=0)  # (n_way, n_way * n_query)
        prob = prob / prob.sum(dim=0, keepdim=True)
        return prob

    def expand(self, x):
        x = x.expand(self.k, *x.shape)
        return x

    def method_cluster_baseline(self, z_all):
        x = self.sub_train_mean(z_all)
        prob_support, prob_query, centers = self.cluster(x)
        x = self.expand(x)
        prob_given_k = self.prob_given_cluster(x, prob_support)
        prob = self.bayesian_prob(prob_given_k, prob_query)
        return prob

    def method_sub_and_cluster(self, z_all, mean_func):
        x = mean_func(z_all)
        prob_support, prob_query, centers = self.cluster(x)
        x = self.expand(x)
        x = self.mean_per_cluster(x, prob_support, prob_query)
        prob_given_k = self.prob_given_cluster(x, prob_support)
        prob = self.bayesian_prob(prob_given_k, prob_query)
        return prob

    def method_proj_and_cluster(self, z_all, mean_func, proj_func):
        x = mean_func(z_all)
        basis = proj_func(x)
        x = self.project_and_norm(x, basis)
        prob_support, prob_query, centers = self.cluster(x)
        x = self.expand(x)
        x = self.mean_per_cluster(x, prob_support, prob_query)
        prob_given_k = self.prob_given_cluster(x, prob_support)
        prob = self.bayesian_prob(prob_given_k, prob_query)
        return prob

    def mean_per_cluster(self, x, prob_support, prob_query):
        """
        :param x: (k, supp + query, n_feat)
        :param prob_support: (k, supp, 1)
        :param prob_query: (k, 1, query)
        :return:
        """
        z_support = x[:, :(self.n_way * self.n_support)]
        z_query = x[:, (self.n_way * self.n_support):]

        z_support_factored = z_support * prob_support
        z_query_factored = z_query * prob_query.transpose(-1, -2)

        center_support = z_support_factored.mean(dim=1, keepdim=True)
        center_query = z_query_factored.mean(dim=1, keepdim=True)

        center_support = F.normalize(center_support, p=2, dim=-1)
        center_query = F.normalize(center_query, p=2, dim=-1)

        z_support = z_support - center_support
        z_query = z_query - center_query

        x = torch.cat((z_support, z_query), dim=1)
        return x
