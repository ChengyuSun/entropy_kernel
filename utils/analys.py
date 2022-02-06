from operator import pos
import sys
sys.path.append('/home/scy/entropy_kernel')
import copy
import os
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes import graph
import numpy as np
from count_motif import allocate
import pandas as pd
from utils.kPCA import pca
from utils.util import read_graph_label
import random
from graphlet.count_graphlet import dataset_reps


GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'

def complete_path(folder, fname):
    return os.path.join(folder, fname)

def ana(dataset_name):
    data = dict()
    dirpath = '/new_disk_B/scy/{}'.format(dataset_name)
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        # print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath,dtype=np.int, delimiter=',')

    graph_ids = set(data['_graph_indicator.txt'])

    node_label_num=max(data[NODE_LABELS_SUFFIX])-min(data[NODE_LABELS_SUFFIX])+1
    print('node labels number: ',node_label_num)

    adj=data[ADJACENCY_SUFFIX]
    edge_index=0
    node_index_begin = 0
    for g_id in set(graph_ids):
        print('正在处理图：'+str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()
        temp_nodN=len(node_ids)
        temp_A=np.zeros([temp_nodN,temp_nodN],int)
        while (edge_index<len(adj))and(adj[edge_index][0]-1 in node_ids):
            e1=adj[edge_index][0]-1-node_index_begin
            e2=adj[edge_index][1]-1-node_index_begin
            temp_A[e1][e2]=temp_A[e2][e1]=1
            edge_index+=1
        node_labels=data[NODE_LABELS_SUFFIX][node_ids]
        node_index_begin += temp_nodN
        node_in_motif=allocate(temp_A)
        


def draw_topology_graph(adj,node_labels,nodN,label,graph_name):
    colors=['red','blue','green','orange','gray','pink','yellow']
    G = nx.Graph()
    G.add_nodes_from([i for i in range(nodN)])
    for j in range(nodN):
        for k in range(nodN):
            if adj[j][k]>0:
                G.add_edge(j,k)

    color_map = []
    for node in G:
        color_map.append(colors[node_labels[node]])

    plt.title(str(label), fontsize=20)
    nx.draw(G, node_color=color_map, with_labels=True,pos=nx.spring_layout(G))
    plt.savefig('./pca_plot/{}.png'.format(graph_name))
    # plt.show()
    return

def update(data):
    shadow = 0
    decay=0.9
    for i in range(len(data)):
        if i==0:
            shadow=data[i]
        else:
            shadow = decay*shadow+(1-decay)*data[i]
            data[i]= shadow
    return data


def draw_subgraph_vs_von():
    graph_num=5976
    date_start=198601
    date_end=201102
    months=['06/1987','01/1990','08/1992','02/1995','09/1997','04/2000','11/2002','06/2005','01/2008','08/2010']
    x=[i for i in range(graph_num)]
    graphlet_num=np.loadtxt('/new_disk_B/scy/financial/graph_reps.txt',np.int)
    
    y_subgraph=update(graphlet_num[:,2])
    y_von=update(np.loadtxt('/home/scy/entropy_kernel/utils/VN.csv'))

    plt.clf()
    figure,(ax1,ax2) = plt.subplots(2,1,
                                    # figsize=(5,6),
                                    # dpi=600,
                                    # 共享x轴
                                    sharex=True)
    s1=ax1.plot(x, # x轴数据
            y_subgraph, # y轴数据
            linestyle = '-', # 折线类型
            color = 'steelblue', # 折线颜色
            label='Subgraph Entropy',
            )
    ax1.set_ylabel('Subgraph Entropy',color = 'steelblue')
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.get_yaxis().get_major_formatter().set_scientific(False)
    ax1.set_xlim(0, graph_num+1)

    # ax2 = ax1.twinx()
    line2=ax2.plot(x, # x轴数据
            -y_von, # y轴数据
            linestyle = '-', # 折线类型
            color = 'seagreen', # 折线颜色
            label='Von Neumann Entropy'
            )
    ax2.set_ylabel('Von Neumann Entropy',color = 'seagreen')
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax2.set_xlim(0, graph_num+1)

    new_ticks = np.linspace(0, graph_num-1, 10)
    plt.xticks(new_ticks,months)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    maxsize = 0.2
    m = 0.2
    s = maxsize / plt.gcf().dpi * graph_num + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    # plt.legend(loc='upper right')
    plt.savefig('./utils/subgraph_vs_von.pdf',bbox_inches='tight')

def drew_subgraphs_financial():
    graph_num=5976
    months=['06/1987','01/1990','08/1992','02/1995','09/1997','04/2000','11/2002','06/2005','01/2008','08/2010']
    x=[i for i in range(graph_num)]
    graphlet_num=np.loadtxt('/new_disk_B/scy/financial/graph_reps.txt',np.int)
    colors=['olive','chocolate','blue']
    plt.clf()
    figure,(ax1,ax2,ax3) = plt.subplots(3,1,
                                    figsize=(12,6),
                                    # dpi=600,
                                    # 共享x轴
                                    sharex=True)
    ax_list=[ax1,ax2,ax3]
    idxs=[1,7,2]
    for i in [0,1,2]:
        y=graphlet_num[:,idxs[i]]
        if i==2:
            y_i=update(y)
        else:
            y_i=-update(y)
        ax_list[i].plot(x, 
                y_i, 
                linestyle = '-',
                linewidth=2,
                color =colors[i],
                # label='graphlet {}'.format(k)
                )
        plt.setp(ax_list[i].get_yticklabels(), visible=False)
        ax_list[i].get_yaxis().get_major_formatter().set_scientific(False)
 
    plt.xlim(0, graph_num+1)
    plt.yticks([])
    new_ticks = np.linspace(0, graph_num-1, 10)
    plt.xticks(new_ticks,months)
    plt.savefig('./utils/all_type_graphlet.pdf',bbox_inches='tight')
    return

drew_subgraphs_financial()


def drew_pca_financial():
    graphlet_num=np.loadtxt('/new_disk_B/scy/financial/graph_reps.txt',np.int)
    N=len(graphlet_num)
    labels=np.zeros((N),int)
    labels[87]=1 #black monday
    labels[86]=2
    # labels[590]=1 #mimi crash
    # labels[580]=2
    # labels[770:920] #early 1990
    # labels[2451:2605]=1 #Asian
    # labels[2815:2930]=2 #russian
    
    # labels[3213:3236]=1 #dot-com
    # labels[3594:3609]=1 #september 11
    # labels[3787:3965]=2 #down turn
    labels[5077:5372]=1 #financial crisis 2007-2008
    # labels[]
    newX=pca(graphlet_num,3)
    colors=['lightgreen','k','r','b','olive']
    alphas=[0.8,1,1]
    ms=[4,8,8,4,4]
    markers=['o','^','v']
    comment=['Other trading days','Black Monday','One day before Black Monday']

    ax = plt.figure().add_subplot(111, projection = '3d')
    p=0.7
    # line_idxs=random.sample(range(len(newX)),k=int(len(newX)*p))
    for line in range(88):
        # print(line)
        x=newX[line][0]
        y=newX[line][1]
        z=newX[line][2]
        l=labels[line]
        if l==0:
            s1=ax.scatter(x, y, z, c=colors[l],alpha=alphas[l],s=ms[l],marker=markers[l])
        if l==1:
            s2=ax.scatter(x, y, z, c=colors[l],alpha=alphas[l],s=ms[l],marker=markers[l])
        if l==2:
            s3=ax.scatter(x, y, z, c=colors[l],alpha=alphas[l],s=ms[l],marker=markers[l])
        
    ax.set_zlabel('3st principle component')
    ax.set_ylabel('2st principle component')
    ax.set_xlabel('1st principle component')

    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_zaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_scientific(False)


    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)

    ax.view_init(elev=10,    # 仰角
                 azim=70    # 方位角
                )
    plt.legend((s1,s2,s3),('Trading days around','Black Monday','One day before Black Monday') ,loc='upper right')
    plt.savefig('./utils/pca_black_monday.pdf')
    
    return


# drew_pca_financial()


def djia():#6346
    d=[]
    label=[]
    d1=pd.read_csv('/home/scy/entropy_kernel/financial_plot/djia.csv',header=None)
    d2=pd.read_csv('/home/scy/entropy_kernel/financial_plot/djia2.csv')
    for i in np.array(d1.iloc[1:,1])[::-1]:
        d.append(i.replace(',',''))
    for j in np.array(d2.iloc[1:,1])[::-1]:
        d.append(j.replace(',',''))
    for i in np.array(d1.iloc[1:,0])[::-1]:
        label.append(i)
    for j in np.array(d2.iloc[1:,0])[::-1]:
        label.append(j)
    return np.array(d).astype(np.float),label

def draw_djia():
    y,x=djia()
    print(len(x))
    x=np.arange(6346)
    plt.plot(x, # x轴数据
                y, # y轴数据
                linestyle = '-', # 折线类型
                color = 'steelblue', # 折线颜色
                )
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    maxsize = 0.3
    m = 0.2
    s = maxsize / plt.gcf().dpi * 6346 + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.savefig('./financial_plot/djia.png')
    
    
def draw_pca_detail(features,labels,dataset,kt=1000,r=3,norm=True,theta1=30,theta2=40):
    if not os.path.exists('./pca_plot_{}'.format(dataset)):
        os.makedirs('./pca_plot_{}'.format(dataset))
    newX=pca(features,3)
    N=len(features)
    colors=['g','b','r']
    ax = plt.figure().add_subplot(111, projection = '3d')
    p=0.7
    line_idxs=random.sample(range(len(newX)),k=int(len(newX)*p))
    for line in line_idxs:
        # if labels[line]==2:
        x=newX[line][0]
        y=newX[line][1]
        z=newX[line][2]
        ax.scatter(x, y, z, c=colors[int(labels[line])],alpha=0.8,marker='^')
        
    ax.set_zlabel('3st principle component')
    ax.set_ylabel('2st principle component')
    ax.set_xlabel('1st principle component')

    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_zaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_scientific(False)


    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)

    
    ax.view_init(elev=theta1,    # 仰角
                 azim=theta2    # 方位角
                )
    # plt.savefig('./pca_plot_{}/{}_{}_{}_{}_{}_{}.png'.format(dataset,dataset,kt,r,norm,theta1,theta2))
    plt.savefig('./utils/mutag.pdf')
    return

def draw_pca_main(dataset):
    labels=read_graph_label(dataset)
    # kt=[0.01,0.1,1,10,100,1000,10000,100000]
    # r=[0.03,0.3,3,30,300,3000,30000]
    kt=[0.01]
    r=[3000000]
    norm=[False]
    # theta=[10,20,30,40,50,60,70,80]
    for i in kt:
        for j in r:
            for k in norm: 
                features=dataset_reps(dataset,i,j,k)
                draw_pca_detail(features,labels,dataset,i,j,k,10,70)

# draw_pca_main('MUTAG')

model_names=['TL-GNN','RW-GNN','DGCNN','DAGCN','GIN','GCN']
data_names=['MUTAG','PTC','NCI1','PROTEINS','COX2','IMDB-B','IMDB-M','Snthetic']
colors=['royalblue','indianred','olive','plum','lightgreen','c']
def draw_performance(diagram_len,diagram_datas,data_name):
    x=np.arange(diagram_len)
    plt.clf()
    plt.figure(figsize=(5,2.5))
    for y in range(5,-1,-1):
        yi=update(diagram_datas[y][:diagram_len])
        # yi=diagram_datas[y][:diagram_len]
        plt.plot(x, 
                yi, 
                linestyle = '-',
                linewidth=1.7,
                color =colors[y],
                label=model_names[y]
                )
    font = {'family' : 'Times New Roman','weight' : 'normal','size' : 8}
    plt.xlabel('Epoch',font,loc='right')
    plt.xlim(0, diagram_len+1)
    plt.ylabel('Accuracy',font,loc='top')
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 8}
    plt.tick_params(labelsize=8)
    plt.legend(loc='lower right',prop=font1)
    plt.savefig('./plot_train/{}.pdf'.format(data_name),bbox_inches='tight')
           
def read_train_performance():
    data_xls = pd.read_excel('/home/scy/entropy_kernel/financial_plot/training_performance.xlsx',header=None,names=None)
    a=np.array(data_xls)
    col_start=2
    for d in data_names:
        diagram_len=800
        diagram_datas=[]
        for m in model_names:
            origin=a[:,col_start][1:].astype(np.float)
            pure=origin[np.where(np.isnan(origin)==False)]
            if len(pure)!=0:
                diagram_len=min(len(pure),diagram_len)
                diagram_datas.append(pure)
            else:
                diagram_datas.append(origin)
            col_start+=1
        print('drawing {},the lenth is {}'.format(d,diagram_len))
        draw_performance(diagram_len,diagram_datas,d)
        col_start+=1
    
# read_train_performance()
# drew_djia()
# draw_graph_financial()

# ana('MUTAG')
# ana('NCI1')
# f=open('/home/scy/PGD/run.sh','w')
# for i in range(1,5976+1):
#     # print(i)
#     # f.write('./pgd -f /new_disk_B/scy/financial/financial_A_{}.txt --micro /new_disk_B/scy/financial/financial_graphlet_{}.txt -o natural --s2l\n'.format(i,i))
#     fin=open('/new_disk_B/scy/financial/financial_graphlet_{}.txt'.format(i))
#     a=fin.readlines()
#     fout=open('/new_disk_B/scy/financial/financial_graphlet_{}.txt'.format(i),'w')
#     b=''.join(a[1:])
#     fout.write(b)
#     fin.close()
#     fout.close()

