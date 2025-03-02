from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import rdkit
from rdkit import Chem
import csv
import os
from tqdm import tqdm
import pickle as cp
from collections import defaultdict
from gln.common.cmd_args import cmd_args
from gln.common.mol_utils import cano_smarts, cano_smiles, smarts_has_useless_parentheses


def process_smiles():
    all_symbols = set() #用于储存所有出现的原子信息

    smiles_cano_map = {}    #用于储存smiles和canonical smiles的对应关系
    for rxn in tqdm(rxn_smiles):
        reactants, _, prod = rxn.split('>')
        mols = reactants.split('.') + [prod]
        for sm in mols: #对于每一个分子sm进行处理
            m, cano_sm = cano_smiles(sm)    #将每个smiles转换为规范形式
            if m is not None:
                for a in m.GetAtoms():
                    all_symbols.add((a.GetAtomicNum(), a.GetSymbol()))  #提取分子中的原子信息储存
            if sm in smiles_cano_map:   #将原始smiles与规范smiles的对应关系储存
                assert smiles_cano_map[sm] == cano_sm
            else:
                smiles_cano_map[sm] = cano_sm
    print('num of smiles', len(smiles_cano_map))
    set_mols = set()
    for s in smiles_cano_map:
        set_mols.add(smiles_cano_map[s])
    print('# unique smiles', len(set_mols))   
    #将 smiles_cano_map 保存为 cano_smiles.pkl 文件。 
    with open(os.path.join(cmd_args.save_dir, 'cano_smiles.pkl'), 'wb') as f:
        cp.dump(smiles_cano_map, f, cp.HIGHEST_PROTOCOL)
    print('# unique atoms:', len(all_symbols))
    all_symbols = sorted(list(all_symbols))
    #将所有出现的原子信息（原子序数）保存为 atom_list.txt 文件
    with open(os.path.join(cmd_args.save_dir, 'atom_list.txt'), 'w') as f:
        for a in all_symbols:
            f.write('%d\n' % a[0])


if __name__ == '__main__':

    raw_data_root = os.path.join(cmd_args.dropbox, cmd_args.data_name)
    rxn_smiles = []
    for phase in ['train', 'val', 'test']:
        csv_file = os.path.join(raw_data_root, 'raw_%s.csv' % phase)
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rxn_idx = header.index('reactants>reagents>production')
            for row in tqdm(reader):
                rxn_smiles.append(row[rxn_idx])

    process_smiles()
    
