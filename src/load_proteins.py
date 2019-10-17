import torch
import numpy as np
from ast import  literal_eval

import networkx as nx
from sklearn.model_selection import StratifiedKFold
class Graph0:
    def __init__(self, graph, nodes):
        self.graph = graph.to('cuda') # Ajacency Matrix
        self.nodes = nodes.to('cuda') # means C, N, ... Br
        self.label = label.to('cuda')

class Graph:
    def __init__(self, nx_graph):
        self.nx_graph = nx_graph
        self.neighbors = []
        self.max_degree = None
        self.node_labels = []
        self.node_features = []
        self.node_attributes = []
        self.edges = []
        self.label = None

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def load_data_table():
    graph_idx_file = '../dataset/PROTEINS/PROTEINS_graph_indicator.txt'
    adjacency_file = '../dataset/PROTEINS/PROTEINS_A.txt'
    node_label_file = '../dataset/PROTEINS/PROTEINS_node_labels.txt'
    node_attribute_file = '../dataset/PROTEINS/PROTEINS_node_attributes.txt'
    graph_label_file = '../dataset/PROTEINS/PROTEINS_graph_labels.txt'

    graph_list = []

    
    graph_size_list = []
    
    cur_g_idx = 0
    cur_size = 0
    graph_list.append(Graph(nx.Graph()))
    with open(graph_idx_file) as fp:
        for line in fp.readlines():
            if int(line) - 1 == cur_g_idx:
                cur_size += 1
                graph_list[cur_g_idx].nx_graph.add_node(cur_size)
            else:
                graph_list.append(Graph(nx.Graph()))
                graph_size_list.append(cur_size)
                cur_g_idx = int(line) - 1
                cur_size = 1
                graph_list[cur_g_idx].nx_graph.add_node(cur_size)
    graph_size_list.append(cur_size)

    assert len(list(graph_list[-1].nx_graph.nodes)) == cur_size


    graph_idx = 0
    pre_sum = 0
    size_sum = graph_size_list[graph_idx] 
    with open(adjacency_file) as fp:
        for line in fp.readlines():
            edge = literal_eval(line)
            if edge[0] > size_sum or edge[1] > size_sum:
                graph_idx += 1
                pre_sum = size_sum
                size_sum += graph_size_list[graph_idx]
            graph_list[graph_idx].nx_graph.add_edge(edge[0] - pre_sum, edge[1] - pre_sum)

        
    assert len(list(graph_list[0].nx_graph.edges)) == 81
    assert len(list(graph_list[1].nx_graph.nodes)) == graph_size_list[1]

    graph_idx = 0
    size = 0
    node_idx = 0
    graph = None
    node_label_map = {}
    with open(node_label_file) as fp1:
        with open(node_attribute_file) as fp2:
            for node_label, node_attribute  in zip(fp1.readlines(), fp2.readlines()):
                if not node_label in node_label_map:
                    mapped_label = len(node_label_map)
                    node_label_map[node_label] = mapped_label
                # 当前graph的node已填满 或者node_idx == size == 0，需初始化size和nodes
                if node_idx == size:
                    graph = graph_list[graph_idx]
                    size = graph_size_list[graph_idx]
                    node_idx = 0
                    graph_idx += 1
                graph.node_labels.append(node_label_map[node_label])
                graph.node_attributes.append(float(node_attribute))
                node_idx += 1
    assert len(graph_list[-1].node_labels) == graph_size_list[-1]

    for i, graph in enumerate(graph_list):
        neighbors = [[] for i in range(len(graph.nx_graph))]
        edges = [[edge[0]-1, edge[1]-1] for edge in graph.nx_graph.edges]
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
        graph.neighbors = neighbors

        degree_list = []
        for i in range(len(graph.nx_graph)):
            degree_list.append(len(neighbors[i]))
        graph.max_degree = max(degree_list)

        edges.extend([[j, i] for i, j in edges])
        graph.edges = torch.LongTensor(edges).transpose(0,1)

        graph.node_features = torch.zeros(len(graph.node_labels), len(node_label_map))
        graph.node_features[range(len(graph.node_labels)), [node_label for node_label in graph.node_labels]] = 1


    label_map = {}
    
    with open(graph_label_file) as fp:
        for graph_idx, graph_label in enumerate(fp.readlines()):
            if not graph_label in label_map:
                mapped_label = len(label_map)
                label_map[graph_label] = mapped_label
            
            graph_list[graph_idx].label = label_map[graph_label]

    return graph_list, len(label_map)


def load_data():
    graph_idx_file = '../dataset/PROTEINS/PROTEINS_graph_indicator.txt'
    adjacency_file = '../dataset/PROTEINS/PROTEINS_A.txt'
    node_label_file = '../dataset/PROTEINS/PROTEINS_node_labels.txt'
    node_attribute_file = '../dataset/PROTEINS/PROTEINS_node_attributes.txt'
    graph_label_file = '../dataset/PROTEINS/PROTEINS_graph_labels.txt'

    adjacency_list = []
    graph_size_list = []

    cur_g_idx = 1
    cur_size = 0

    with open(graph_idx_file) as fp:
        for line in fp.readlines():
            if int(line) == cur_g_idx:
                cur_size += 1
            else:
                adjacency_list.append(torch.zeros(cur_size, cur_size))
                graph_size_list.append(cur_size)
                cur_size = 1
                cur_g_idx = int(line)
    adjacency_list.append(torch.zeros(cur_size, cur_size))
    graph_size_list.append(cur_size)

    # print(graph_size_list[:3])  # [42, 27, 10]
    size_sum = 0

    graph_idx = 0
    pre_sum = 0
    size_sum += graph_size_list[graph_idx] 
    with open(adjacency_file) as fp:
        for line in fp.readlines():
            edge = literal_eval(line)
            if edge[0] > size_sum or edge[1] > size_sum:
                graph_idx += 1
                pre_sum = size_sum
                size_sum += graph_size_list[graph_idx]
            adjacency_list[graph_idx][edge[0] - 1 - pre_sum, edge[1] - 1 - pre_sum] = 1.0

    # print(graph_list[0].sum()) # tensor(162.)

    dim = 4
    graph_idx = 0
    size = 0
    node_idx = 0
    node_list = []
    with open(node_label_file) as fp1:
        with open(node_attribute_file) as fp2:
            for node_label, node_attribute  in zip(fp1.readlines(), fp2.readlines()):
                # 当前graph的node已填满 或者node_idx == size == 0，需初始化size和nodes
                if node_idx == size:
                    size = graph_size_list[graph_idx]
                    nodes = torch.zeros(size, dim)
                    node_idx = 0
                    graph_idx += 1
                
                node_label = int(node_label)
                node_attribute = float(node_attribute)
                node_val = 1.0
                if node_label == 0:
                    nodes[node_idx] = torch.tensor([node_val, 0.0, 0.0, node_attribute])
                elif node_label == 1:
                    nodes[node_idx] = torch.tensor([0.0, node_val, 0.0, node_attribute])
                else:
                    nodes[node_idx] = torch.tensor([0.0, 0.0, node_val, node_attribute])

                node_idx += 1

                if node_idx == size:
                    node_list.append(nodes)

    # print(node_list[0].shape)  # torch.Size([42, 4])

    graph_list = []
    label_list = []
    
    with open(graph_label_file) as fp:
        for graph_idx, graph_label in enumerate(fp.readlines()):
            
            graph_label = torch.tensor([float(graph_label) - 1.0])
            label_list.append(graph_label)
            # if graph_label == '1':
            #     label_list.append(torch.tensor([1.0, 0.0]).to('cuda'))
            # else:
            #     label_list.append(torch.tensor([0.0, 1.0]).to('cuda'))

            graph_list.append(Graph0(adjacency_list[graph_idx], node_list[graph_idx], graph_label))
        
    return graph_list, label_list
    
# import os               
if __name__ == "__main__":
    # test loading graph_idx_file

    _, classes = load_data_table()
    


    # os.chdir('DeepGraphFramework/GraphIsomorphismNetwork/src')
    # molecule_list, label_list = load_data()    # data includes Ajacency and node_info       

    # molecule0 = molecule_list[0]    

    # print(molecule0.nodes)
    # print(molecule0.graph[0, :])
    # print(torch.mm(molecule0.graph, molecule0.nodes))
                

                






        

