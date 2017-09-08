#!/usr/bin/env python
# encoding: utf-8
# This is the class wrapper around dataloader.
import numpy as np
import scipy.sparse as sp
import os
import sklearn.datasets as sd
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity


# wrapper for load_svmlight_file in dealing with empty files / no-existing files
def load_svmlight_file(f, n_features=None, dtype=np.float64, multilabel=False, zero_based='auto', query_id=False):
    if not os.path.exists(f) or (os.stat(f).st_size == 0):
        assert n_features is not None
        if multilabel:
            y = [()]
        else:
            y = []
        return sp.csr_matrix((0, n_features)), y
    else:
        return sd.load_svmlight_file(f, n_features=n_features, dtype=dtype, multilabel=multilabel, zero_based=zero_based, query_id=query_id)


class DataClass:
    # attributes:

    # path: srcPath, tgtPath, prlPath
    # feature dimension: source_n_features, target_n_features
    # kernel: source_gamma, target_gamma
    # kernel_type: 'rbf', 'cosine'
    # valid_flag: use .val.libsvm for grid search, or use .tst.libsvm for reporting final results
    # source_data_type:
    #   'full': use src.full.trn.libsvm as the training data in the source domain, leaving test and para data to be empty
    #   'normal': use both train, test, para data in the source domain
    #   'parallel': use only parallel in the source domain
    # zero_diag_flag: (True) zero-out the diagonal, (False) keep the diagonal to be 1s
    # kernel_normal:
    #   True: use symmetrically normalized W
    #   False: directly use W
    def __init__(self, srcPath=None, tgtPath=None, prlPath=None, valid_flag=True, zero_diag_flag=False, source_data_type='full',
                 source_n_features=100000, target_n_features=200000, kernel_type='cosine', kernel_normal=False,
                 source_gamma=None, target_gamma=None):
        assert(srcPath is not None)
        assert(tgtPath is not None)
        assert(prlPath is not None)
        self.srcPath = srcPath
        self.tgtPath = tgtPath
        self.prlPath = prlPath
        self.source_n_features = source_n_features
        self.target_n_features = target_n_features
        self.kernel_type = kernel_type
        self.kernel_normal = kernel_normal
        self.valid_flag = valid_flag
        self.zero_diag_flag = zero_diag_flag
        self.source_data_type = source_data_type
        if source_gamma is None:
            self.source_gamma = 1.0 / np.sqrt(source_n_features)
        if target_gamma is None:
            self.target_gamma = 1.0 / np.sqrt(target_n_features)

    def kernel(self, data, **parameters):
        if self.kernel_type == 'cosine':
            return cosine_similarity(data)
        elif self.kernel_type == 'rbf':
            return rbf_kernel(data, gamma=parameters['gamma'])

    @staticmethod
    def reduce_para(y, I, K, offset, num_para):
        ''' reduce the number of parallel data, given y, I, K, offset, num_para
        return a 4-element array (y, I, K, offset)'''
        assert len(offset) == 6
        # index to delete
        del_index = range(offset[1] + num_para, offset[2])
        del_index.extend(range(offset[4] + num_para, offset[5]))

        # update offset
        diff_num_para = offset[2] - offset[1] - num_para
        new_offset = offset.copy()
        for i in xrange(2, 5):
            new_offset[i] -= diff_num_para
        new_offset[5] -= 2 * diff_num_para
        # update K
        new_K = np.delete(K, del_index, axis=0)
        new_K = np.delete(new_K, del_index, axis=1)
        # update y, I
        new_y = np.delete(y, del_index)
        new_I = np.delete(I, del_index)

        return new_y, new_I, new_K, new_offset

    ''' do the statitcs on the number of non-zero entries (for each block in K)
     return a 4-element array (#source-source, #source-target, #target-source, #target-target)'''
    @staticmethod
    def sparse_K_stat(K, offset):
        assert len(offset) == 6
        K_ind = (K != 0)
        K_ss = K_ind[:offset[2], :offset[2]]
        K_st = K_ind[:offset[2], offset[2]:]
        K_ts = K_ind[offset[2]:, :offset[2]]
        K_tt = K_ind[offset[2]:, offset[2]:]
        return [K_ss.sum(), K_st.sum(), K_ts.sum(), K_tt.sum()]

    @staticmethod
    def normalize(K):
        inv_sqrt_row_sum = np.diag( 1.0 / np.sqrt(K.sum(axis=1)) )
        return inv_sqrt_row_sum.dot(K).dot(inv_sqrt_row_sum)

    # keep the `nn` nearest neighbor for each row
    # nn is # nearest neighbor to keep
    @staticmethod
    def sparsify_K(K, nn):
        ret_K = np.zeros(K.shape)
        for i in range(K.shape[0]):
            index = np.argsort(K[i, :])[-nn:]
            ret_K[i, index] = K[i, index]
        return ret_K

    # keep the `nn` nearest neighbor for each row, and make the kernel symmetrical by averaging through its transpose
    # nn is # nearest neighbor to keep
    @staticmethod
    def sym_sparsify_K(K, nn):
        K_sp = DataClass.sparsify_K(K, nn)
        K_sp = (K_sp + K_sp.T) / 2  # in case of non-positive semi-definite
        return K_sp

    # Load the data
    # return values:
    # y: (ns+nt): true_values
    # I: (ns+nt) observation indicator
    # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # offset: [source_train , ... , target_para] offset number
    def get_TL_Kernel(self):
        if self.source_data_type == 'normal':
            source_train = self.srcPath + '.trn.libsvm'
            source_test = self.srcPath + '.val.libsvm'  # val is for tuning hyperparameters, in final should report on tst
        elif self.source_data_type == 'full':
            # if srcPath ends with numbers, trail it
            fields = self.srcPath.split('.')
            if fields[-1].isdigit():
                srcPath = '.'.join(fields[:-1])
            else:
                srcPath = self.srcPath
            source_train = srcPath + '.full.trn.libsvm'
            source_test = srcPath + '.full.val.libsvm'  # val is for tuning hyperparameters, in final should report on tst
        elif self.source_data_type == 'parallel':
            source_train = '/non/existent/file'
            source_test = '/non/existent/file'
        else:
            raise ValueError('Unknown source domain data option.')
        source_para = self.prlPath + 'prlSrc.libsvm'

        target_train = self.tgtPath + '.trn.libsvm'
        target_test = self.tgtPath + '.val.libsvm'
        target_para = self.prlPath + 'prlTgt.libsvm'
        if self.valid_flag == False:
            target_test = self.tgtPath + '.tst.libsvm'
        y, I, K, offset = self._get_TL_Kernel(source_train, source_test, source_para,
                                              target_train, target_test, target_para)
        # # normalize y (-1 -> 1)
        # def to_0(x):
        #     if x == -1:
        #         return 0
        #     else:
        #         return x
        # y = np.asarray(map(to_0, y))
        return y, I, K, offset

    def _get_TL_Kernel(self, source_train, source_test, source_para,
                       target_train, target_test, target_para):
        # source_domain, target_domain dimension should be fixed

        source_train_X, source_train_y = load_svmlight_file(source_train, n_features=self.source_n_features)
        source_test_X, source_test_y = load_svmlight_file(source_test, n_features=self.source_n_features)
        source_para_X, _ = load_svmlight_file(source_para, n_features=self.source_n_features, multilabel=True)

        target_train_X, target_train_y = load_svmlight_file(target_train, n_features=self.target_n_features)
        target_test_X, target_test_y = load_svmlight_file(target_test, n_features=self.target_n_features)
        target_para_X, _ = load_svmlight_file(target_para, n_features=self.target_n_features, multilabel=True)
        ##### default gamma value is taken to be sqrt of the data dimension
        ##### May need to tune and change the calculation of
        # if source_gamma == None:
        #     source_gamma = 1.0 / np.sqrt(source_train_X.shape[1])
        # if target_gamma == None:
        #     target_gamma = 1.0 / np.sqrt(target_train_X.shape[1])

        if source_test_X.shape[0] == 0:
            source_data = sp.vstack([source_train_X, source_para_X])
        else:
            source_data = sp.vstack([source_train_X, source_test_X, source_para_X])
        # source_ker = rbf_kernel(source_data, gamma=self.source_gamma)
        source_ker = self.kernel(source_data, gamma=self.source_gamma)

        if target_train_X.shape[0] == 0:
            target_data = sp.vstack([target_test_X, target_para_X])
        else:
            target_data = sp.vstack([target_train_X, target_test_X, target_para_X])
        # target_ker = rbf_kernel(target_data, gamma=self.target_gamma)
        target_ker = self.kernel(target_data, gamma=self.target_gamma)

        len_X = [source_train_X.shape[0] , source_test_X.shape[0] , source_para_X.shape[0]
                , target_train_X.shape[0] , target_test_X.shape[0] , target_para_X.shape[0]]
        offset = np.cumsum(len_X)

        # K initialize
        n = offset[5]
        K = np.zeros([n, n])
        # source/target, kernel
        K[0:offset[2],0:offset[2]] = source_ker
        K[offset[2]:offset[5],offset[2]:offset[5]] = target_ker
        # parallel data
        K[offset[1]:offset[2],offset[4]:offset[5]] = np.diag([1.] * len_X[2])
        K[offset[4]:offset[5],offset[1]:offset[2]] = np.diag([1.] * len_X[2])
        if self.zero_diag_flag:
            np.fill_diagonal(K, 0.0)
        if self.kernel_normal:
            K = DataClass.normalize(K)

        # observation Indicator
        I = np.zeros(n, dtype=np.float)
        I[0:offset[0]] = np.ones(len_X[0], dtype=np.float)
        I[offset[2]:offset[3]] = np.ones(len_X[3], dtype=np.float)

        # true values
        y = np.zeros(n, dtype=np.float)
        y[0:offset[0]] = source_train_y
        y[offset[0]:offset[1]] = source_test_y
        y[offset[2]:offset[3]] = target_train_y
        y[offset[3]:offset[4]] = target_test_y
        y = y.astype(np.int)

        return y, I, K, offset

    # Load the target [train,test] data (no parallel)
    # return values:
    # y: (ns+nt): true_values
    # I: (ns+nt) observation indicator
    # K: (ns+nt) * (ns+nt), basic kernel, which could also be the i~j indicator
    # offset: [train, test] offset
    def get_SSL_Kernel(self):
        target_train = self.tgtPath + '.trn.libsvm'
        target_test = self.tgtPath + '.val.libsvm'
        if self.valid_flag == False:
            target_test = self.tgtPath + '.tst.libsvm'

        target_train_X, target_train_y = load_svmlight_file(target_train, n_features=self.target_n_features)
        target_test_X, target_test_y = load_svmlight_file(target_test, n_features=self.target_n_features)

        # print(type(target_train_X), type(target_train_y), type(target_test_X))
        len_X = [target_train_X.shape[0] , target_test_X.shape[0]]
        offset = np.cumsum(len_X)

        n = offset[1]

        target_data = sp.vstack([target_train_X, target_test_X])
        # target_ker = rbf_kernel(target_data, gamma=self.target_gamma)
        target_ker = self.kernel(target_data, gamma=self.target_gamma)

        y = np.zeros(n, dtype=np.float)
        y[0:offset[0]] = target_train_y
        y[offset[0]:offset[1]] = target_test_y
        y = y.astype(np.int)

        I = np.zeros(n, dtype=np.float)
        I[0:offset[0]] = np.ones(len_X[0], dtype=np.float)

        K = target_ker
        if self.zero_diag_flag:
            np.fill_diagonal(K, 0.0)
        if self.kernel_normal:
            K = DataClass.normalize(K)

        # # normalize y (-1 -> 1)
        # def to_0(x):
        #     if x == -1:
        #         return 0
        #     else:
        #         return x
        # y = np.asarray(map(to_0, y))
        return y, I, K, offset

    @staticmethod
    def complete_TL_Kernel(K_ori, offset):
        # source_train, source_test, source_para
        #            0,           1,           2
        # target_train, target_test, target_para
        #            3,           4,           5
        K = K_ori.copy()
        # complete target parallel
        K[offset[4]:offset[5], 0:offset[1]] = K[offset[1]: offset[2], 0:offset[1]]
        K[offset[2]:offset[4], offset[1]:offset[2]] = K[offset[2]: offset[4], offset[4]:offset[5]]
        K[offset[4]:offset[5], offset[1]:offset[2]] = (K[offset[1]: offset[2], offset[1]:offset[2]] +  K[offset[4]: offset[5], offset[4]:offset[5]]) / 2
        # zero_based should impose 1 to parallel diagonal elements
        xcoos = range(offset[4], offset[5])
        ycoos = range(offset[1], offset[2])
        K[xcoos, ycoos] = 1.0

        K[0:offset[2], offset[2]:offset[5]] = K[offset[2]:offset[5], 0:offset[2]].transpose()
        return K


    def get_TL_Kernel_completeOffDiag(self):
        y, I, K, offset = self.get_TL_Kernel()
        K = DataClass.complete_TL_Kernel(K, offset)
        return y, I, K, offset

# for testing
if __name__ == '__main__':
    dirPath = '/usr0/home/wchang2/research/NIPS2016/data/cls/cls-acl10-postprocess/en_music_de_dvd/'
    srcPath = dirPath + 'src.256'
    tgtPath = dirPath + 'tgt.256'
    prlPath = dirPath
    source_n_features = 60244
    target_n_features = 185922

    #  source_train = srcPath + '.trn.libsvm'
    #  source_test = srcPath + '.tst.libsvm'
    #  source_para = dirPath + 'prlSrc.libsvm'

    #  target_train = tgtPath + '.trn.libsvm'
    #  target_test = tgtPath + '.tst.libsvm'
    #  target_para = dirPath + 'prlTgt.libsvm'

    dc = DataClass()
    dc = DataClass(srcPath=srcPath, tgtPath=tgtPath, prlPath=prlPath,
                    valid_flag=False, zero_diag_flag=False, source_data_type='full',
                    source_n_features=source_n_features, target_n_features=target_n_features,
                    kernel_type='cosine', kernel_normal=False)
    y, I, K, offset = dc.get_TL_Kernel()
    print len(y), len(I), K.shape, offset
    print K[:10, :10]

    y, I, K, offset = DataClass.reduce_para(y, I, K, offset, 100)
    print len(y), len(I), K.shape, offset
    print K[:10, :10]
    # print K[1,2], K[1000, 999], K[1891,K.shape[1]-1]
    # print sum(I)
    # print sum(y)
