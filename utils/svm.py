path='../lib/libsvm-3.24/python'
import sys
sys.path.append(path)
from svmutil import *
from graphlet.count_graphlet import dataset_graph_reps
from utils.util import read_graph_label


dataset='MUTAG'
train_pixel=dataset_graph_reps(dataset).tolist()
print('train_pixel:',len(train_pixel))
print('train_pixel[0]:',len(train_pixel[0]))
train_label=read_graph_label(dataset).tolist()

#train_label,train_pixel = svm_read_problem('../lib/libsvm-3.24/heart_scale')

model = svm_train(train_label[:200],train_pixel[:200],'-c 4')
print("result:")
p_label, p_acc, p_val = svm_predict(train_label[200:], train_pixel[200:], model)
print(p_acc)