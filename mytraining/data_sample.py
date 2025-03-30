import os
import csv
import numpy as np
import random
from gln.data_process.data_info import DataInfo, load_train_reactions
from mytraining.my_cmd_args import cmd_args

class DataSample(object):
    def __init__(self, prod, reaction=None, neg_reactions=None):
        self.prod = prod  # 产物的规范SMILES表示
        self.reaction = reaction  # 规范化后的整条反应
        self.neg_reactions = neg_reactions  # 负样本的反应（可选）


def _rand_sample_except(candidates, exclude, k=None):   #从候选列表candidates中随机选择k个样本，排除指定的exclude项
    assert len(candidates)
    if k is None:
        if len(candidates) == 1:
            assert exclude is None or candidates[0] == exclude
            return candidates[0]
        else:
            while True:
                c = np.random.choice(candidates)
                if exclude is None or c != exclude:
                    break
            return c
    else:        
        if k <= 0 or len(candidates) <= k:
            return [c for c in candidates if exclude is None or c != exclude]
        cand_indices = np.random.permutation(len(candidates))[:k]        
        selected = []
        for i in cand_indices:
            c = candidates[i]
            if exclude is None or c != exclude:
                selected.append(c)
            if k <= 0:
                continue
            if len(selected) >= k:
                break
        return selected


DataInfo.init(cmd_args.dropbox, cmd_args)
args = cmd_args

seed = np.random.randint(10000)
np.random.seed(seed)
random.seed(seed)
num_epochs = 0
part_id = 0
train_reactions = load_train_reactions(args)
while True:
    if num_epochs % args.epochs_per_part == 0:  # args.epochs_per_part = 1，每隔一定轮数加载数据分区
        DataInfo.load_cooked_part('train', part_id)
        tot_num = len(train_reactions)
        part_size = tot_num // args.num_parts + 1
        indices = range(part_id * part_size, min((part_id + 1) * part_size, tot_num))
        indices = list(indices)
        part_id = (part_id + 1) % args.num_parts
    random.shuffle(indices)
    for sample_idx in indices:
        rxn_type, rxn_smiles = train_reactions[sample_idx]
        reactants, _, prod = rxn_smiles.split('>')
        cano_prod = DataInfo.smiles_cano_map[prod]

        sample = DataSample(prod=cano_prod)
        # 整个规范化后的反应
        sample.reaction = DataInfo.get_cano_smiles(reactants) + '>>' + cano_prod
        sample.neg_reactions = []
        if len(DataInfo.neg_reactions_all[sample_idx]):
            neg_reacts = DataInfo.neg_reactions_all[sample_idx]
            if len(neg_reacts):
                neg_reactants = _rand_sample_except(neg_reacts, None, args.neg_num)
                sample.neg_reactions = [DataInfo.neg_reacts_list[r] + '>>' + cano_prod for r in neg_reactants]