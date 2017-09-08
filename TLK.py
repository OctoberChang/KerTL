#!/usr/bin/env python
# encoding: utf-8
# This file tries to do semi-supervised learning on target domain test data
# using all data [source_train, source_test, source_para, target_train, target_test, target_para].
# We do not complete `K`.
# The solution for prediction `f` is exact.
# The tuning parameters `w_2` for coefficient for regularization term. (we use default gamme, which is the sqrt of dimension for each domain)

import scipy
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from dataclass import DataClass
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import cvxopt
from cvxopt import matrix

np.random.seed(123)


def eval_binary(y_true, y_prob):
    unique_class = np.unique(y_true)
    assert(len(unique_class) == 2)
    if np.min(y_true) == -1:
        y_true = (y_true + 1) / 2.0

    n_pos = y_true[y_true == 1].shape[0]
    y_pred = np.zeros(y_true.shape[0])
    y_pred[np.argsort(y_prob)[::-1][:n_pos]] = 1
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    return auc, acc


def cvxopt_solver(y, I, K, offset, w_2):
    # closed form
    n = y.shape[0]
    D = np.diag(np.sum(K, axis=1))
    lap = D - K

    P = lap * w_2 + np.diag(I)
    q = -I * y
    G = -np.diag(np.ones(n))
    h = np.zeros(n)
    # using cvxopt quadratic programming:
    #    min_x  1/2 xTPx + qTx
    #    s.t.   Gx <= h
    #           Ax = b
    # reference: https://github.com/cvxopt/cvxopt
    #            http://cvxopt.org/examples/
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['feastol'] = 1e-6
    sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    f = np.array(sol['x'])[:, 0]

    # for calculating ap
    start_offset = offset[3]
    end_offset = offset[4]

    y_true = y[start_offset:end_offset]
    y_prob = f[start_offset:end_offset]
    return y_true, y_prob


def eigen_decompose(K, offset, max_k=None):
    W_s = K[:offset[2], :offset[2]]
    W_t = K[offset[2]:, offset[2]:]
    if max_k is None:
        v_s, Q_s = scipy.linalg.eigh(W_s)
        v_t, Q_t = scipy.linalg.eigh(W_t)
    else:
        v_s, Q_s = sp.linalg.eigsh(W_s, k=max_k)
        v_t, Q_t = sp.linalg.eigsh(W_t, k=max_k)

    return v_s, Q_s, v_t, Q_t


def get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t, beta):
    K_exp = K.copy()
    Y_st = K_exp[:offset[2], offset[2]:]
    Lambda_s = np.diag(np.exp(beta * v_s))
    Lambda_t = np.diag(np.exp(beta * v_t))
    K_ss = Q_s.dot(Lambda_s.dot(Q_s.T))
    K_tt = Q_t.dot(Lambda_t.dot(Q_t.T))
    K_st = K_ss.dot(Y_st.dot(K_tt))
    K_st = normalize(K_st, norm='l2', axis=1)
    K_exp[:offset[2], offset[2]:] = K_st
    K_exp[offset[2]:, :offset[2]] = K_st.T
    return K_exp


def run_one(srcPath=None, tgtPath=None, prlPath=None, prlSize=None,
            source_n_features=None, target_n_features=None, kernel_type='cosine'):

    dc = DataClass(srcPath=srcPath, tgtPath=tgtPath, prlPath=prlPath,
                   valid_flag=False, zero_diag_flag=False, source_data_type='full',
                   source_n_features=source_n_features, target_n_features=target_n_features,
                   kernel_type=kernel_type, kernel_normal=False)
    y, I, K, offset = dc.get_TL_Kernel()
    y, I, K, offset = DataClass.reduce_para(y, I, K, offset, prlSize)

    # run eigen decomposition on K
    v_s, Q_s, v_t, Q_t = eigen_decompose(K, offset, max_k=128)
    beta = 2**(-10)
    wreg = 2**(-10)
    K_exp = get_K_exp_by_eigen(K, offset, v_s, Q_s, v_t, Q_t, beta)
    K_exp[K_exp<0] = 0
    y_true, y_prob = cvxopt_solver(y, I, K_exp, offset, wreg)
    auc, acc = eval_binary(y_true, y_prob)
    return auc, acc
