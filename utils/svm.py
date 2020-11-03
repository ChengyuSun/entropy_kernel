path='../lib/libsvm-3.24/python'
import sys
sys.path.append(path)
from svmutil import *
from graphlet.count_graphlet import dataset_reps
from utils.util import read_graph_label,acc_calculator
import numpy as np
import random
from sklearn import svm
from utils.kPCA import js_kernel_process


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

    #print('test_idx from {} to {}'.format(test_start,test_end))
    return np.array(train_idx),np.array(test_idx)


def kernel_svm(kernel_values,train_idx,test_idx):
    print(len(kernel_values[train_idx].tolist()[0]))
    prob = svm_problem(labels[train_idx].tolist(), kernel_values[train_idx].tolist(), isKernel=True)
    param = svm_parameter('-t 4 -c 4 -b 1')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(labels[test_idx].tolist(), kernel_values[test_idx].tolist(), model)
    return p_acc[0]

def normal_svm(train_idx,test_idx):
    model = svm_train(labels[train_idx].tolist(), features[train_idx].tolist(), '-c 4')
    print("result:")
    p_label, p_acc, p_val = svm_predict(labels[test_idx].tolist(), features[test_idx].tolist(), model)
    return p_acc[0]

def sklearn_svm(train_idx,test_idx):
    clf_linear = svm.SVC(kernel='linear')
    clf_linear.fit(features[train_idx], labels[train_idx])
    score_linear = clf_linear.score(features[test_idx], labels[test_idx])
    #print("The score of linear is : %f" % score_linear)
    return score_linear*100

accs=[]
n=10
kernel_features=js_kernel_process(features,8)
for i in range(n):
    train_idx_temp, test_idx_temp = n_cross(n,i, nodN, random_idx)
    #accs.append(sklearn_svm(train_idx_temp,test_idx_temp))
    accs.append(kernel_svm(kernel_features,train_idx_temp,test_idx_temp))
print(dataset)
acc_calculator(accs)