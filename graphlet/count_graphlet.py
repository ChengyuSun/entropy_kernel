import copy
import numpy as np
# def find_chain(adj,N,start,chain_lenth,stack):
#     if chain_lenth==0:
#         adj[start].fill(0)
#         for j in range(N):
#             adj[j][start] = 0
#         return start
#     else:
#         if np.sum(adj[start])>0:
#             x = np.nonzero(adj[start])
#             for j in range(N):
#                 adj[j][start]=0
#             adj[start].fill(0)
#             next_Index=x[0][0]
#             stack.append(next_Index)
#             return find_chain(adj,N,next_Index,chain_lenth-1,stack)
#         else:
#             stack.pop()
#             return -1
#     return
#
# def count_label_graphlet_chain(begin_index,chain_lenth,Adj,N):
#     a=copy.copy(Adj)
#     for i in range(N):
#         stack=[]
#         stack.append(a)
#         if find_chain()
#     return

def to_decimalism(base_system,number_vactor):  #将base_system的编号序列转化为十进制
    res=0
    number_vactor.reverse()
    for index,value in enumerate(number_vactor):
        res+=value*pow(base_system,index)
    return res


def graphlet_diffuse(start_index,adj_original,N,node_labels,label_num):
    graphlet_index_2 = pow(label_num,2)
    graphlet_index_3_1 = graphlet_index_2+pow(label_num,3)
    graphlet_index_3_2 = graphlet_index_3_1+pow(label_num,3)
    graphlet_index_3_3 = graphlet_index_3_2 + pow(label_num,3)
    graphlet_index_4_1=graphlet_index_3_3+pow(label_num,4)
    graphlet_index_4_2=graphlet_index_4_1+pow(label_num,4)
    graphlet_index_4_3=graphlet_index_4_2+pow(label_num,4)
    graphlet_index_4_4=graphlet_index_4_3+pow(label_num,4)
    graphlet_types=graphlet_index_4_4-1
    node_rep=[0 for i in graphlet_types]

    neighbors_1=np.nonzero(adj_original[start_index])[0]
    for n1 in neighbors_1:
        number_vactor=[]
        number_vactor.append(node_labels[start_index])



    return

def generate_graphlet_index(label_num):

    return