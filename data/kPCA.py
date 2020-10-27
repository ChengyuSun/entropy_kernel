#encoding:utf-8
import numpy as np
import csv
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

#gamma: a free parameter for the RBF kernel
#k : the number of components to be returned 
def rbf_kpca(X, gamma, k):     
    #Calculating the squared Euclidean distances for every pair of points
    #in the M*N dimensional dataset
    sq_dist = pdist(X, metric='sqeuclidean')
                            # N = X.shape[0]
                            # sq_dist.shape = N*(N-1)/2


    #Converting the pairwise distances into a symmetric M*M matirx
    mat_sq_dist = squareform(sq_dist)
                            # mat_sq_dist.shape = (N, N)
    

    #Computing the M*M kernel matrix
    # step 1
    K = np.exp(-gamma*mat_sq_dist)


    #Computing the symmetric N*N kernel matrix
    # step 2
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    Lambda=np.real(Lambda)
    Q=np.real(Q)
    #print(Lambda)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    eigen_pairs = sorted(eigen_pairs, reverse=True, key=lambda k: k[0])
    return np.column_stack((eigen_pairs[i][1] for i in range(k)))


def meanX(dataX):
    return np.mean(dataX,axis=0)


def pca(origin_matrix,target_dim):
    pca = PCA(n_components=target_dim)
    pca.fit(origin_matrix)
    newX = pca.fit_transform(origin_matrix)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    return newX


def js_kernel(v1,v2):
    dim=len(v1.tolist())
    kl_divergence=0
    js_divergence=0
    for i in range(dim):
    #todo

        return