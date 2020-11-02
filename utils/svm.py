path='../lib/libsvm-3.24/python'
import sys
sys.path.append(path)
from svmutil import *
from graphlet.count_graphlet import dataset_reps
from utils.util import read_graph_label,acc_calculator
import numpy as np
import random


dataset='MUTAG'
features=dataset_reps(dataset)
labels=read_graph_label(dataset)

nodN=len(labels.tolist())
random_idx = [i for i in range(nodN)]
random.shuffle(random_idx)

def n_cross(n,index,nodN,random_idx):

    test_start=index*(nodN//n)
    test_end=test_start+(nodN//n)
    train_idx = random_idx[:test_start]+random_idx[test_end:]
    test_idx = random_idx[test_start:test_end]
    train_idx = train_idx[:test_start]+train_idx[test_end:]

    return np.array(train_idx),np.array(test_idx)


def kernel_svm(labels,kernel_values):
    prob = svm_problem(labels, kernel_values, isKernel=True)
    param = svm_parameter('-t 4 -c 4 -b 1')
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(train_label[test_idx].tolist(), train_pixel[test_idx].tolist(), model)
    return p_acc[0]

def normal_svm():
    model = svm_train(labels[train_idx].tolist(), features[train_idx].tolist(), '-c 4')
    print("result:")
    p_label, p_acc, p_val = svm_predict(labels[test_idx].tolist(), features[test_idx].tolist(), model)
    return p_acc[0]

accs=[]
n=10
for i in range(n):
    train_idx, test_idx = n_cross(n,i, nodN, random_idx)
    model = svm_train(train_label[train_idx].tolist(), train_pixel[train_idx].tolist(), '-c 4')
    print("result:")
    p_label, p_acc, p_val = svm_predict(train_label[test_idx].tolist(), train_pixel[test_idx].tolist(), model)
    accs.append(p_acc[0])
print(dataset)
acc_calculator(accs)