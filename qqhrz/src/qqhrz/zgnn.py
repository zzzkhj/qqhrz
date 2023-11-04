import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GraphSAGE(nn.Module):
    """采用邻居"""
    def __init__(self, sk: list[int], node_features, num_hiddens, out_features, agg_method='sum', agg_hidden='concat'):
        """
        采样邻居更新点的向量。
        :param sk:每层采用的个数
        :param node_features:节点的特征维度
        :param num_hiddens:隐层的维度
        :param out_features:输出维度
        :param agg_method:对采样的邻居聚合的方式，sum、mean和max，默认sum
        :param agg_hidden:聚合后的邻居节点特征和这个节点特征聚合的方式，concat和sum，默认concat
        """
        super().__init__()
        self.sk = sk
        self.A = None
        self.agg_method = agg_method
        self.agg_hidden = agg_hidden
        self.sample_nodes = None
        self.dense1 = nn.Linear(node_features, num_hiddens)
        self.dense2 = nn.Linear(num_hiddens * 2, num_hiddens)
        self.ln = nn.LayerNorm(num_hiddens)
        self.dense3 = nn.Linear(num_hiddens, out_features)

    def forward(self, X, edge_index):
        # 得到邻接矩阵
        num_nodes = X.shape[0]
        if self.A is None:
            self.A = torch.zeros(num_nodes, num_nodes)
            self.A[edge_index[0], edge_index[1]] = 1
            self.A += torch.eye(num_nodes)
            self.A.to(X.device)

        if self.sample_nodes is None:
            self.sample_nodes = self.get_k_subgraph()

        X = self.dense1(X)
        num_update_nodes = 1
        for i, k in enumerate(self.sk):
            num_update_nodes *= k
            if self.agg_method == 'sum':
                sum_sample_nodes = X[self.sample_nodes[i]].reshape(num_nodes, num_update_nodes, -1).sum(dim=1)
            elif self.agg_method == 'mean':
                sum_sample_nodes = X[self.sample_nodes[i]].reshape(num_nodes, num_update_nodes, -1).mean(dim=1)
            elif self.agg_method == 'max':
                sum_sample_nodes = X[self.sample_nodes[i]].reshape(num_nodes, num_update_nodes, -1).max(dim=1).values
            else:
                raise ValueError(f'agg_method属性不存在为{self.agg_method}的值。只能在sum、mean和max中选择，默认是sum。')
            if self.agg_hidden == 'concat':
                X = torch.cat([X, sum_sample_nodes], dim=1)
                X = self.dense2(X)
            elif self.agg_hidden == 'sum':
                X = X + sum_sample_nodes
            else:
                raise ValueError(f'agg_hidden属性不存在为{self.agg_hidden}的值。只能在concat和sum中选择，默认是concat。')
            X = self.ln(F.relu(X))
        return self.dense3(X)

    def sample(self, nodes, s):
        all_select_nodes = []
        for node in nodes:
            select_nodes = torch.nonzero(self.A[node]).reshape(-1)
            all_select_nodes.append(torch.tensor(random.choices(select_nodes, k=s)))
        return torch.cat(all_select_nodes, dim=0)

    def get_k_subgraph(self):
        sample_nodes = [torch.arange(self.A.shape[0])]
        for i, s in enumerate(self.sk):
            sample_nodes.append(self.sample(sample_nodes[i], s))
        return sample_nodes[1:]
