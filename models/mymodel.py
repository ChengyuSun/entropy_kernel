from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import keras
from graphlet.count_graphlet import dataset_reps
from utils.util import read_graph_label

def get_feature_and_label(dataset):
    original_features=dataset_reps(dataset)
    original_labels=read_graph_label(dataset)

    N=len(original_features)
    d=len(original_features[0])
    print("N: ",N)
    print("d: ",d)

    for i in range(N):
        if original_labels[i]==-1:
            original_labels[i]=0

    return original_features,original_labels,N,d

def build_model(input_dim):
    model = Sequential()

    model.add(Dense(units=1,  input_dim=input_dim))

    # model.add(Dense(units=32, activation='relu', input_dim=d))
    # model.add(Dense(units=16, activation='softmax'))
    # model.add(Dense(units=8, activation='softmax'))
    # model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model

def train(model,features_train,labels_train,N):
    model.fit(features_train[N//10:], labels_train[N//10:], epochs=5, batch_size=32)
    #model.train_on_batch(x_batch, y_batch)

def evaluate(model,features_test,labels_test):
    loss,acc = model.evaluate(features_test[:N//10], labels_test[:N//10], batch_size=128)
    return acc

def predict(model,input_features):
    classes = model.predict(input_features, batch_size=128)
    return classes