# path='../lib/libsvm-3.24/python'
from enum import Flag
from locale import normalize
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# print( sys.path)
# sys.path.append('../')
from graphlet.count_graphlet import dataset_reps,dataset_reps_financial
from utils.util import read_graph_label,acc_calculator,write_ppi, write_ptc
import numpy as np
import random
from sklearn import svm
from utils.kPCA import pca
# from thundersvm import SVC
import os


def n_cross(n,index,nodN,random_idx):

    test_start=index*(nodN//n)
    test_end=test_start+(nodN//n)
    train_idx = random_idx[:test_start]+random_idx[test_end:]
    test_idx = random_idx[test_start:test_end]
    return np.array(train_idx),np.array(test_idx)


# def kernel_svm(kernel_values,labels,train_idx,test_idx):
#     prob = svm_problem(labels[train_idx].tolist(), kernel_values[train_idx].tolist(), isKernel=True)
#     param = svm_parameter('-t 4 -c 4 -b 1')
#     model = svm_train(prob, param)
#     p_label, p_acc, p_val = svm_predict(labels[test_idx].tolist(), kernel_values[test_idx].tolist(), model)
#     return p_acc[0]

# def normal_svm(features,labels,train_idx,test_idx):
#     model = svm_train(labels[train_idx].tolist(), features[train_idx].tolist(), '-c 4')
#     p_label, p_acc, p_val = svm_predict(labels[test_idx].tolist(), features[test_idx].tolist(), model)
#     return p_acc[0]

def sklearn_svm(kernel,features,labels,train_idx,test_idx):
    kernel='poly' if kernel=='polynomial' else kernel
    clf_linear = svm.SVC(kernel=kernel)
    clf_linear.fit(features[train_idx], labels[train_idx])
    score_linear = clf_linear.score(features[test_idx], labels[test_idx])
    return score_linear*100

def gpu_svm(k,features,labels,train_idx,test_idx):
    model = SVC(kernel=k)
    model.fit(features[train_idx], labels[train_idx])
    score_linear = model.score(features[test_idx], labels[test_idx])
    return score_linear*100

def ten_ten_svm(features,labels,kernel,graph_num):
    accs = []
    for k in range(10):
        temp_accs = []
        random_idx = [i for i in range(graph_num)]
        random.shuffle(random_idx)
        for i in range(10):
            train_idx_temp, test_idx_temp = n_cross(10, i, graph_num, random_idx)
            print('ready for {}:{}'.format(k,i))
            # temp_score=gpu_svm(kernel,features,labels,train_idx_temp,test_idx_temp)
            temp_score=sklearn_svm(kernel,features,labels,train_idx_temp,test_idx_temp)

            print('{}:{}  score is {}'.format(k,i,temp_score))
            temp_accs.append(temp_score)
            #temp_accs.append(kernel_svm(kernel_features, original_labels,train_idx_temp, test_idx_temp))
        temp_res,dis=acc_calculator(temp_accs)
        print('\n------10-fold-result: {} -------\n'.format(format(temp_res,'.3f')))
        accs.append(temp_res)
    return accs


dataset='PROTEINS'
kernel=['polynomial']
#'linear','polynomial','sigmoid','rbf']
graphlet_normalize=[False]
# log_base=[0,2,4,6,8,10,100,1000]
kt_list=[10000]#0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000
r_list=[0.1]
log_list=[0.02,0.2,2,5,10,15,20,50,100,500,1000]#0.02,0.2,2,5,10,15,20,50,100,500,1000
labels=read_graph_label(dataset)

for kt in kt_list:
    for r in r_list:
        for log_value in log_list:
            for normal_j in graphlet_normalize: 
                with open('./log_PROTEINS.txt','a') as log_file:
                    log_file.write('dataset:{}  KT:{}  r:{}  log:{}  normalize:{} \n'.format(dataset,kt,r,log_value,normal_j))
                features=dataset_reps(dataset,kt,r,log_value,normal_j)
                for kernel_i in kernel:
                    graph_num=len(labels.tolist())
                    # for i in range(graph_num):
                    #     print('f:{}  label:{}'.format(features[i],labels[i]))
                    # sys.exit()
                    accs=ten_ten_svm(features,labels,kernel_i,graph_num=graph_num)
                    print('\n\n\n--------------------------\n'
                            '----------average----------\n'
                            '----------results------------\n'
                            '---------of-10-10-folds---------\n'
                            '---------------------------\n')
                    avg,dis=acc_calculator(accs)
                    with open('./log_PROTEINS.txt','a') as log_file:
                        log_file.write('\t kernel:{}  avg:{:.3f}   dis:{:.3f}\n'.format(kernel_i,avg,dis))