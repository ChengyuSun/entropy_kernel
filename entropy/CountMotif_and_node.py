#encoding: utf-8
import numpy as np
import csv
import copy
# number of motif

Nm = 8

def clean_node(a,index):
    a[index].fill(0)
    a[:,index].fill(0)

def count_star(A,N,neiN):#todo
    n=0
    a=copy.copy(A)
    for i in range(N):
        if (np.sum(a[i])>neiN-1):
            n+=1
            for j in range(i):
                a[N-j-1][i]=0
            x=np.nonzero(a[i])
            nei_Index=x[0][:neiN]
            a[i].fill(0)
            for j in nei_Index:
                a[j].fill(0)
                for k in range(N):
                    a[k][j]=0
    return n

def find_next(a,N,i,rest,stack):
    if rest==0:
        return 1

    next_index_list = np.nonzero(a[i])[0]

    for next_index in next_index_list:
        if next_index not in stack:
            stack.append(next_index)
            if find_next(a, N, next_index, rest - 1, stack) > 0:
                return 1

    stack.pop(a)
    return -1

def count_chain(A,N,len,node_occupation,motif):
    n=0
    a = copy.copy(A)
    for i in range(N):
        stack = [i]
        if find_next(a,N,i,len-1,stack)>0:
            print('find chain ',stack)
            for j in stack:
                node_occupation[j] += str(motif)
                clean_node(a,j)
            n+=1
    return n

def count_triangle(A, N,node_occupation):
    n=0
    a = copy.copy(A)
    for i in range(N):
        for j in range(i,N):
            if a[i][j]>0:
                for k in range(j,N):
                    if a[j][k]>0 and a[k][i] >0:

                        n+=1
                        node_occupation[i] += str(3)
                        node_occupation[j] += str(3)
                        node_occupation[k] += str(3)
    return n

def count_quadrangle(A, N,node_occupation):#todo
    n=0
    a = copy.copy(A)
    for i in range(N):
        for j in range(i,N):
            if a[i][j]>0:
                for k in range(j,N):
                    if a[j][k]>0:
                        for l in range(k,N):
                            if a[k][l]>0 and a[l][i]>0:
                                n+=1
                                node_occupation[i] += str(6)
                                node_occupation[j] += str(6)
                                node_occupation[k] += str(6)
                                node_occupation[l] += str(6)

    return n




def count_Motifs(A):
    nodN = len(A)
    node_occupation = ['' for i in range(nodN)]
    rd = np.argsort(sum(np.transpose(A)))
    rdA = A[rd]
    rdA[:, ] = rdA[:, rd]

    Nm_1 = count_chain(rdA, nodN, 2, node_occupation, 1)
    Nm_2 = count_chain(rdA, nodN, 3, node_occupation, 2)
    #todo
    Nm_3 = count_triangle(rdA, 3)
    Nm_4 = count_chain(rdA, nodN, 4, node_occupation, 4)
    Nm_5 = count_star(rdA, nodN, 3, node_occupation)
    Nm_6 = count_quadrangle(rdA, 4, node_occupation)
    Nm_7 = count_chain(rdA, nodN, 5, node_occupation, 7)
    Nm_8 = count_star(rdA, nodN, 4, node_occupation)
    num = [Nm_1, Nm_2, Nm_3, Nm_4, Nm_5, Nm_6, Nm_7, Nm_8]
    return num

