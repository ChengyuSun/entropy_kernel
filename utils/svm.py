path='../lib/libsvm-3.24/python'
import sys
sys.path.append(path)
from svmutil import *
from graphlet.count_graphlet import dataset_graph_reps
from utils.util import read_graph_label
import numpy as np
import random


dataset='NCI1'
train_pixel=dataset_graph_reps(dataset)
train_label=read_graph_label(dataset)

nodN=len(train_label.tolist())
random_idx = [i for i in range(nodN)]
random.shuffle(random_idx)

def ten_cross(index,nodN,random_idx):

    test_start=index*(nodN//10)
    test_end=test_start+(nodN//10)
    train_idx = random_idx[:test_start]+random_idx[test_end:]
    test_idx = random_idx[test_start:test_end]
    train_idx = train_idx[:test_start]+train_idx[test_end:]

    return np.array(train_idx),np.array(test_idx)


#train_label,train_pixel = svm_read_problem('../lib/libsvm-3.24/heart_scale')
acc_sum=0
for i in range(10):
    train_idx, test_idx = ten_cross(i, nodN, random_idx)
    model = svm_train(train_label[train_idx].tolist(), train_pixel[train_idx].tolist(), '-c 4')
    print("result:")
    p_label, p_acc, p_val = svm_predict(train_label[test_idx].tolist(), train_pixel[test_idx].tolist(), model)
    acc_sum+=p_acc[0]

print(dataset+' : '+str(acc_sum/10))