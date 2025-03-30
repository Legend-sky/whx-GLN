import numpy as np
import pandas as pd
import os
from rdkit import Chem
import random
import csv
from mytraining.my_cmd_args import cmd_args
from gln.data_process.data_info import DataInfo, load_center_maps
from mytraining.my_model_inference import RetroGLN
from gln.common.evaluate import get_score, canonicalize

from tqdm import tqdm
import torch

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

import argparse
cmd_opt = argparse.ArgumentParser(description='Argparser for test only')
cmd_opt.add_argument('-model_for_test', default='mytraining/mytraining/LocalRetro_train/model-900.dump', help='model for test')
local_args, _ = cmd_opt.parse_known_args()

def exact_match(preds, true):
    for k, pred in enumerate(preds):
        try:
            if pred == true:
                return k+1
        except Exception as e:
            pass
    return -1

def demap(mols, stereo = True):
    if type(mols) == type((0, 0)):
        ss = []
        for mol in mols:
            [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, stereo))
            if mol == None:
                return None
            ss.append(Chem.MolToSmiles(mol))
        return '.'.join(sorted(ss))
    else:
        [atom.SetAtomMapNum(0) for atom in mols.GetAtoms()]
        return '.'.join(sorted(Chem.MolToSmiles(mols, stereo).split('.')))

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    launch_folder = os.getcwd() # 获取当前工作目录 /home/dell/whx/whx-GLN
    test_file = pd.read_csv('data/uspto_50k/raw_test.csv')
    rxn_prod = [rxn.split('>>')[1] for rxn in test_file['reactants>reagents>production']]

    ground_truth = []
    for rxn in test_file['reactants>reagents>production']:
        ground_truth.append(demap(Chem.MolFromSmiles(rxn.split('>>')[0])))

    results = {}
    local_scores = {}
    result_file = 'data/LocalRetro_USPTO_50K.txt'
    with open(result_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split('\n')[0]
            i = int(line.split('\t')[0])
            predictions = line.split('\t')[1:]
            results[i] = [eval(p)[0] for p in predictions]
            local_scores[i] = [eval(p)[1] for p in predictions]

    model = RetroGLN(cmd_args.dropbox, local_args.model_for_test)
    print('testing', local_args.model_for_test)

    Exact_matches = []
    for i, prod in enumerate(rxn_prod):
        prod = canonicalize(prod)
        reacts = results[i]
        scores, list_reacts = model.my_run(prod, reacts)

        # scores = scores + local_scores[i]

        # 创建排序键：每个元素是一个元组 (score, original_index)
        keys = [(score, idx) for idx, score in enumerate(scores)]

        # 对排序键进行排序，同时保留相同分数的原始顺序
        sorted_keys = sorted(keys, key=lambda x: (-x[0], x[1]))

        # 根据排序后的键重新排列反应式列表
        sorted_reacts = [list_reacts[idx] for (_, idx) in sorted_keys]
        match_exact = exact_match(sorted_reacts, ground_truth[i])
        Exact_matches.append(match_exact)
        if i % 100 == 0:
            print ('\rCalculating accuracy... %s/%s' % (i, len(results)), end='', flush=True)
    
    ks = [1, 3, 5, 10, 50]
    exact_k = {k:0 for k in ks}
    print(len(Exact_matches))
    for i in range(len(Exact_matches)):
        for k in ks:
            if Exact_matches[i] <= k and Exact_matches[i] != -1:
                exact_k[k] += 1

    for k in ks:
        print ('Top-%d Exact accuracy: %.3f' % (k, exact_k[k]/len(Exact_matches)))
