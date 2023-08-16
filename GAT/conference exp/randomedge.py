from dgl.nn.pytorch import GATConv
from dgl.data import CoraGraphDataset
import dgl
from dgl.data.utils import load_graphs
from copy import deepcopy
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd


def load_cora_data():
    data = CoraGraphDataset()
    graph = data[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    #mask = graph.ndata['train_mask']
    return graph, features, labels #, mask

def train_test_split(labels):
    for_test = []
    test_size = int(labels.shape[0] * 0.2) # train 0.8, test 0.2
    train_mask = torch.tensor([True for i in range(labels.shape[0])])
    test_mask = torch.tensor([False for i in range(labels.shape[0])])
    wanna_mask = torch.randint(0, labels.shape[0], (test_size, ))
    for i in wanna_mask:
        index = i.item()
        train_mask[index] = False
        test_mask[index] = True
    return train_mask, test_mask

"""====================================================================================================================
    Load graph_list.bin
    return a graph list sotre with dgl.graphs()
    _                         
    | |__   ___  __ _ _ __ ___ 
    | '_ \ / _ \/ _` | '__/ __|
    | |_) |  __/ (_| | |  \__ \
    |_.__/ \___|\__,_|_|  |___/
===================================================================================================================="""
def Load_GraphList():

    graph_list= load_graphs('graph_list.bin') # graph list, graph label
    graph_list = graph_list[0]
    return graph_list

"""
    Slicing list into chunks, each chunk has n elements.
"""
def chunks(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))

"""
    Random shuffle a list
"""
def shuffle_chunks(input, batch_num=10):
    chunk = deepcopy(input)
    chunk_size = float(len(chunk)/batch_num)
    if chunk_size - int(chunk_size):
        chunk_size += 1
    chunk_size = int(chunk_size)

    #   dataset random slicing into 10 chunks 
    random.shuffle(chunk)
    chunk = list(chunks(xs= chunk, n= chunk_size))
    return chunk

""" ====================================================================================================================
    Isolated Vertex Process
        U = {u | for all u is top 10 degrees node of graph}
        for v in isolated(graph):
            add edge of u->v in this graph
            
====================================================================================================================ʕ •ᴥ•ʔ """ 
# find isolated vertex from Graph
def Isolated_V(graph):

    in_iso = []
    out_iso = []
    for i in ((graph.in_degrees() == 0)).nonzero():
        in_iso.append(i.item())

    for i in ((graph.out_degrees() == 0)).nonzero():
        out_iso.append(i.item())
        
    isolated = set(in_iso).intersection(set(out_iso))
    return list(isolated)

# isolated process
def isolated_process(isolated, data_g):

    graph = deepcopy(data_g)
    # find top 10 degrees
    top10_deg = sorted(graph.out_degrees(), reverse=True) 
    top10_deg = top10_deg[0:10]
    top10_deg = [i.item() for i in top10_deg]

    top_deg_nodes = []
    for degree in top10_deg:
        node = (graph.out_degrees() == degree).nonzero()
        for i in node:
            top_deg_nodes.append(i.item())
     # find nodes by top degrees
    for v in isolated:
        u = random.choice(top_deg_nodes)
        graph.add_edges(u,v)
    return graph

"""
    Split Postive and Negative set of the input dataset
"""
def PN_split(ind: list, label):
    p = list()
    n = list()
    for idx in ind:
        if label[idx] == 0:
            n.append(idx)
        else:
            p.append(idx)
    return p, n

"""
    Negative Sampling
        return |Positive samples| *3 's Negative Samples with random choices
"""
def Negative_Sampling(pos_sample, neg_samples):
    NS = list()
    if len(neg_samples)/len(pos_sample) < 3:
        NS = deepcopy(neg_samples)
    else:
        NS = random.sample(neg_samples, len(pos_sample)*3)
    return NS

"""
    Process base graph , features, labels into formats and shape of new data graph
"""
def Data_Graph_Process(del_nodes, nodes_set, g, feature, active_matrix, label):
    data_g = deepcopy(g)
    data_g = dgl.remove_nodes(data_g, del_nodes)
    datag_f = feature[nodes_set]   # data graph feature matrix
    datag_label = label[nodes_set]     # data graph label
    datag_label = datag_label.to(torch.long)
    datag_active_matrix = active_matrix[nodes_set]
    iso_d = Isolated_V(graph= data_g)   # isolated nodes of data graph
    data_g = isolated_process(isolated= iso_d, data_g= data_g)
    return data_g, datag_f, datag_label, datag_active_matrix, iso_d

"""
    Split training and testing set
"""
def train_test_split(train_set, test_set, nodes):
    train_mask = torch.zeros(len(nodes), dtype= torch.bool)
    test_mask = torch.zeros(len(nodes), dtype= torch.bool)
    node_list = sorted(nodes)
    encode_table = dict()
    count = 0

    for k in node_list:
        encode_table[k] = count
        count += 1

    for train_sample in train_set:
        idx = encode_table[train_sample]
        train_mask[idx] = True
        
    for test_sample in test_set:
        idx = encode_table[test_sample]
        test_mask[idx] = True

    return train_mask, test_mask

"""
    Saving loss value plots
"""
def plotting(x: list, y: list, file_path: str, title: str):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.plot(x, y)
    plt.savefig(file_path + title + '.png')
    plt.clf()

"""
    Print Confusion Matrix 4 numbers into file
"""
def confusion_to_file(tp, tn, fp, fn, file_pointer):
    print('     TP :', tp, file= file_pointer)
    print('     TN :', tn, file= file_pointer)
    print('     FP :', fp, file= file_pointer)
    print('     FN :', fn, file= file_pointer)


"""
    Write Model's results and the confusion matrix to excel files
"""
def write_to_pd(test, path, batch_id):
    column_name = ['epoch', 'train loss', 'TP', 'TN', 'FP', 'FN']
    col = list()
    for i in test:
        col.append(i)
    df = pd.DataFrame(col, columns= column_name)
    name = 'Batch %d' % batch_id
    if batch_id == 1:
        df.to_excel(path + 'result.xlsx', index= False, sheet_name= name)
    else:
        with pd.ExcelWriter(path + 'result.xlsx', engine= 'openpyxl', mode= 'a') as writer:
            df.to_excel(writer, sheet_name= name, index= False)