#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import os
import numpy as np
from TLK import run_one

dataPath = '/usr0/home/yuexinw/research/mnist/view1_view2/'
kernel_type='rbf'
tgtSize= 2
dimDict = {'src': 392, 'tgt': 392}
digitList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nr_fold = 5
for c in digitList:
    auc_result = []
    for fold in range(nr_fold):
        dirPath = dataPath + 'digit%d/fold%d/' % (c, fold)
        assert(os.path.isdir(dirPath))
        srcPath = dirPath + 'src.1024'
        tgtPath = dirPath + 'tgt.%d' % (tgtSize)
        prlPath = dirPath
        auc, acc = run_one(srcPath=srcPath, tgtPath=tgtPath,
                           prlPath=prlPath, prlSize=1024,
                           source_n_features=dimDict['src'],
                           target_n_features=dimDict['tgt'],
                           kernel_type=kernel_type)
        auc_result.append(auc)

    auc_result = np.array(auc_result)
    auc_mean = np.mean(auc_result)
    auc_std = np.std(auc_result)
    print('digit %d: ' % (c), end='')
    print('\t %f/%f' % (auc_mean, auc_std))
