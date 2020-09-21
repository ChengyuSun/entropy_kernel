import copy
import numpy as np

class GraphletCoder:
    label_number=0
    graphlet_index_2 =0
    graphlet_index_3_1 = 0
    graphlet_index_3_2 = 0
    graphlet_index_3_3 = 0
    graphlet_index_4_1 = 0
    graphlet_index_4_2 = 0
    graphlet_index_4_3 = 0
    graphlet_index_4_4 = 0
    graphlet_types = 0

    def __init__(self, label_num):
        self.graphlet_index_2 = pow(label_num, 2)
        self.graphlet_index_3_1 = self.graphlet_index_2 + pow(label_num, 3)
        self.graphlet_index_3_2 = self.graphlet_index_3_1 + pow(label_num, 3)
        self.graphlet_index_3_3 = self.graphlet_index_3_2 + pow(label_num, 3)
        self.graphlet_index_4_1 = self.graphlet_index_3_3 + pow(label_num, 4)
        self.graphlet_index_4_2 = self.graphlet_index_4_1 + pow(label_num, 4)
        self.graphlet_index_4_3 = self.graphlet_index_4_2 + pow(label_num, 4)
        self.graphlet_index_4_4 = self.graphlet_index_4_3 + pow(label_num, 4)
        self.graphlet_types = self.graphlet_index_4_4 - 1
        self.graphlet_type_vactor = [0,self.graphlet_index_2, self.graphlet_index_3_1, self.graphlet_index_3_2,
                                     self.graphlet_index_3_3,
                                     self.graphlet_index_4_1, self.graphlet_index_4_2, self.graphlet_index_4_3]

        self.label_number=label_num
        return

    def code(self,number_vactor,graphlet_type):
        res = 0
        number_vactor.reverse()
        for index, value in enumerate(number_vactor):
            res += value * pow(self.label_number, index)
        return res+self.graphlet_type_vactor[graphlet_type]

    def get_graphlet_types(self):
        return self.graphlet_types



def graphlet_diffuse(start_index,adj_original,node_labels,graphlet_coder):

    node_rep=[0 for i in graphlet_coder.get_graphlet_types()]

    neighbors_1=np.nonzero(adj_original[start_index])[0]
    for index_1_1,n1_1 in enumerate(neighbors_1):
        #2
        number_vactor_2=[node_labels[start_index],node_labels[n1_1]]
        node_rep[graphlet_coder.code(number_vactor_2,0)]+=1

        #3_1
        neighbors_2=np.nonzero(adj_original[n1_1])[0].remove(start_index)
        for index_2_1,n2_3 in enumerate(neighbors_2):
            number_vactor_3_1 = [start_index, n1_1, n2_3]
            node_rep[graphlet_coder.code(number_vactor_3_1, 1)] += 1
            for n2_4 in neighbors_2[index_2_1+1:]:
                #4_2
                number_vactor_4_2 = [start_index, n1_1, n2_3,n2_4]
                node_rep[graphlet_coder.code(number_vactor_4_2, 5)] += 1
            for n3_1 in np.nonzero(adj_original[n2_3])[0].remove(start_index).remove(n1_1):
                #4_1
                number_vactor_4_1 = [start_index, n1_1, n2_3, n3_1]
                node_rep[graphlet_coder.code(number_vactor_4_1, 4)] += 1


        for index_1_2,n1_2 in enumerate(neighbors_1[index_1_1+1:]):
            #3_3
            if adj_original[n1_1][n1_2]==1:
                number_vactor_3_3=[start_index,n1_1,n1_2]
                node_rep[graphlet_coder.code(number_vactor_3_3,3)]+=1
            else :
                #3_2
                number_vactor_3_2=[start_index,n1_1,n1_2]
                node_rep[graphlet_coder.code(number_vactor_3_2, 2)] += 1
                #4_3
                for n1_3 in neighbors_1[index_1_2 + 1:]:
                    number_vactor_4_3 = [start_index, n1_1, n1_2, n1_3]
                    node_rep[graphlet_coder.code(number_vactor_4_3, 6)] += 1
                #4_4
                for n2_1 in np.nonzero(adj_original[n1_1])[0]:
                    if n2_1!=start_index and n2_1!=n1_2:
                        number_vactor_4_4 = [start_index, n1_1, n1_2, n2_1]
                        node_rep[graphlet_coder.code(number_vactor_4_4, 7)] += 1
                for n2_2 in np.nonzero(adj_original[n1_2])[0]:
                    if n2_2 != start_index and n2_2 != n1_1:
                        number_vactor_4_4 = [start_index, n1_2, n1_1, n2_2]
                        node_rep[graphlet_coder.code(number_vactor_4_4, 7)] += 1
    return node_rep

def generate_graphlet_index(label_num):

    return