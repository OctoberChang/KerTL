#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import os
import numpy as np
from TLK import run_one

dataPath = '/usr0/home/yuexinw/bonda/research/cls/'
kernel_type='cosine'
tgt = ['de', 'fr', 'jp']
tgtSize= 2
domain = ['books', 'dvd', 'music']
dimDict = {'en': 60244, 'de': 185922, 'fr': 59906, 'jp': 75809}
seed_list = [0, 1, 2]
for d_s in domain:
    for t in tgt:
        for d_t in domain:
            if d_s == d_t:
                continue
            auc_result = []
            for seed in seed_list:
                dirPath = dataPath + 'cls_seed_%d/en_%s_%s_%s/' % (seed, d_s, t, d_t)
                assert(os.path.isdir(dirPath))
                srcPath = dirPath + 'src.1024'
                tgtPath = dirPath + 'tgt.%d' % (tgtSize)
                prlPath = dirPath
                auc, acc = run_one(srcPath=srcPath, tgtPath=tgtPath,
                                   prlPath=prlPath, prlSize=1024,
                                   source_n_features=dimDict['en'],
                                   target_n_features=dimDict[t],
                                   kernel_type=kernel_type)
                auc_result.append(auc)

            auc_result = np.array(auc_result)
            auc_mean = np.mean(auc_result)
            auc_std = np.std(auc_result)
            print('en_%s_%s_%s (tgt=2): ' % (d_s, t, d_t), end='')
            print('\t %f/%f' % (auc_mean, auc_std))
