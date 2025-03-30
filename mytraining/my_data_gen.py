from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import random
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from multiprocessing import Process, Queue
import time

from gln.data_process.data_info import DataInfo, load_train_reactions
from gln.common.reactor import Reactor


class DataSample(object):
    def __init__(self, prod, reaction=None, neg_reactions=None):
        self.prod = prod  # 产物的规范SMILES表示
        # self.center = center  # 反应中心的规范SMARTS表示
        # self.template = template  # 反应模板
        # self.label = label  # 标签（可选）
        # self.neg_centers = neg_centers  # 负样本的反应中心
        # self.neg_tpls = neg_tpls  # 负样本的反应模板
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


def worker_softmax(worker_id, seed, args):
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
            if len(sample.neg_reactions):
                yield (worker_id, sample)
        num_epochs += 1


def worker_process(worker_func, worker_id, seed, data_q, *args):
    worker_gen = worker_func(worker_id, seed, *args)
    for t in worker_gen:
        data_q.put(t)


def data_gen(num_workers, worker_func, worker_args, max_qsize=16384, max_gen=-1, timeout=60):
    cnt = 0
    data_q = Queue(max_qsize)

    if num_workers == 0:  # single process generator
        worker_gen = worker_func(-1, np.random.randint(10000), *worker_args)
        while True:
            worker_id, data_sample = next(worker_gen)
            yield data_sample
            cnt += 1
            if max_gen > 0 and cnt >= max_gen:
                break
        return

    worker_procs = [Process(target=worker_process, args=[worker_func, i, np.random.randint(10000), data_q] + worker_args) for i in range(num_workers)]
    for p in worker_procs:
        p.start()
    last_update = [time.time()] * num_workers    
    while True:
        if data_q.empty():
            time.sleep(0.1)
        if not data_q.full():
            for i in range(num_workers):
                if time.time() - last_update[i] > timeout:
                    print('worker', i, 'is dead')
                    worker_procs[i].terminate()
                    while worker_procs[i].is_alive():  # busy waiting for the stop of the process
                        time.sleep(0.01)
                    worker_procs[i] = Process(target=worker_process, args=[worker_func, i, np.random.randint(10000), data_q] + worker_args)
                    print('worker', i, 'restarts')
                    worker_procs[i].start()
                    last_update[i] = time.time()            
        try:
            sample = data_q.get_nowait()
        except:
            continue
        cnt += 1
        worker_id, data_sample = sample
        last_update[worker_id] = time.time()
        yield data_sample
        if max_gen > 0 and cnt >= max_gen:
            break

    print('stopping')
    for p in worker_procs:
        p.terminate()
    for p in worker_procs:
        p.join()
