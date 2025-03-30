from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from gln.common.consts import DEVICE, t_float
from torch_scatter import scatter_max, scatter_add, scatter_mean
from gln.graph_logic.graph_feat import get_gnn
from mytraining.my_soft_logic import OnehotEmbedder, CenterProbCalc, ActiveProbCalc, ReactionProbCalc

import torch
from gln.mods.torchext import jagged_log_softmax

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gln.data_process.data_info import DataInfo

# 单独测试反应物的分数训练
class GraphPath(nn.Module):
    def __init__(self, args):
        super(GraphPath, self).__init__()
        self.reaction_predicate = ReactionProbCalc(args)
        # self.reaction_predicate.prod_enc.w_n2l.weight = nn.Parameter(torch.randn(128, 39))  # 根据训练时的维度调整
        # self.reaction_predicate.react_enc.react_gnn.w_n2l.weight=nn.Parameter(torch.randn([128, 39]))  # 根据训练时的维度调整

    def forward(self, samples):
        prods = []
        list_of_list_reacts = []

        for sample in samples:
            prods.append(SmilesMols.get_mol_graph(sample.prod))
               
            list_reacts = [sample.reaction] + sample.neg_reactions
            list_of_list_reacts.append(list_reacts)

        react_log_prob = self.reaction_predicate(prods, list_of_list_reacts)
        loss = - torch.mean(react_log_prob)

        return loss
