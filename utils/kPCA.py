# encoding:utf-8
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import math


# gamma: a free parameter for the RBF kernel
# k : the number of components to be returned
def rbf_kpca(X, gamma, k):
    # Calculating the squared Euclidean distances for every pair of points
    # in the M*N dimensional dataset
    sq_dist = pdist(X, metric='sqeuclidean')
    # N = X.shape[0]
    # sq_dist.shape = N*(N-1)/2

    # Converting the pairwise distances into a symmetric M*M matirx
    mat_sq_dist = squareform(sq_dist)
    # mat_sq_dist.shape = (N, N)

    # Computing the M*M kernel matrix
    # step 1
    K = np.exp(-gamma * mat_sq_dist)

    # Computing the symmetric N*N kernel matrix
    # step 2
    N = X.shape[0]
    one_N = np.ones((N, N)) / N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # step 3
    Lambda, Q = np.linalg.eig(K)
    Lambda = np.real(Lambda)
    Q = np.real(Q)
    # print(Lambda)
    eigen_pairs = [(Lambda[i], Q[:, i]) for i in range(len(Lambda))]
    eigen_pairs = sorted(eigen_pairs, reverse=True, key=lambda k: k[0])
    return np.column_stack((eigen_pairs[i][1] for i in range(k)))


def meanX(dataX):
    return np.mean(dataX, axis=0)


def pca(origin_matrix, target_dim):
    pca = PCA(n_components=target_dim)
    pca.fit(origin_matrix)
    newX = pca.fit_transform(origin_matrix)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    return newX


def js_kernel(v1, v2):
    kl_divergence_1 = 0
    kl_divergence_2 = 0
    for i in range(len(v1)):
        kl_divergence_1 += v1[i] * math.log(v1[i] / v2[i], 2)
        kl_divergence_2 += v2[i] * math.log(v2[i] / v1[i], 2)
    js_divergence = kl_divergence_1 + kl_divergence_2
    return js_divergence / 2


def js_kernel_level(v1, v2, level):
    for k in range(len(v1)):
        v1[k] = math.log(v1[k] + 2, 2)
        v2[k] = math.log(v2[k] + 2, 2)
    v1_new = []
    v2_new = []
    label_num = len(v1) // 8
    for i in range(label_num):
        v1_new += v1[i * 8:i * 8 + level]
        v2_new += v2[i * 8:i * 8 + level]
    sum1 = sum(v1_new)
    sum2 = sum(v2_new)
    for k in range(len(v1_new)):
        v1_new[k] = v1_new[k] / sum1
        v2_new[k] = v2_new[k] / sum2
    return js_kernel(v1_new, v2_new)


def js_kernel_process(input, level=8):
    sample_num = len(input)
    matrix = np.zeros((sample_num, sample_num), float)
    for i in range(sample_num):
        for j in range(sample_num):
            matrix[i][j] = js_kernel_level(input[i].tolist(), input[j].tolist(), level)
    id_vactor = np.arange(1,sample_num+1).reshape(sample_num, 1)
    # print('id_vactor: ',id_vactor.shape)
    # print('matrix: ', matrix.shape)
    matrix = np.append(id_vactor, matrix,axis=1)

    return matrix


def liner_kernel(v1,v2):
    sum=0
    for i in range(len(v1)):
        sum+=v1[i]*v2[i]
    return sum

def linear_kernel_process(input):
    sample_num = len(input)
    matrix = np.zeros((sample_num, sample_num), float)
    for i in range(sample_num):
        for j in range(sample_num):
            matrix[i][j] = liner_kernel(input[i].tolist(), input[j].tolist())
    id_vactor = np.arange(1,sample_num+1).reshape(sample_num, 1)
    # print('id_vactor: ',id_vactor.shape)
    # print('matrix: ', matrix.shape)
    matrix = np.append(id_vactor, matrix,axis=1)

    return matrix