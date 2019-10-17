import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
class GraphIsomorphismNetwork0(object):
    def __init__(self, node_dim, update_loop_size):
        self.node_dim = node_dim
        self.update_loop_size = update_loop_size
    
    def mlp(self, molecule):
        # 经历一步特征迭代
        return molecule.nodes + torch.mm(molecule.graph, molecule.nodes)

    def readout(self, molecule):
        return molecule.nodes.sum(dim=0)

    def predict(self, molecule):
        tmp_molecule = copy.deepcopy(molecule)
        sum_of_nodes = torch.zeros(self.node_dim).to('cuda')
        # CONCAT(READOUT(molecule.nodes at k) k < update_loop_size)
        for i in range(self.update_loop_size):
            # 经历i步特征迭代
            tmp_molecule.nodes = self.mlp(tmp_molecule)
            sum_of_nodes += self.readout(tmp_molecule)
        
        return sum_of_nodes

class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_drop_out, learn_eps, neighbor_aggregating_type, graph_pooling_type, device):
        super(GraphIsomorphismNetwork, self).__init__()
        self.num_layers = num_layers
        self.final_drop_out = final_drop_out
        self.neighbor_aggregating_type = neighbor_aggregating_type
        self.graph_pooling_type = graph_pooling_type
        
        self.learn_eps = learn_eps
        self.device = device

        self.mlps = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.eps = nn.Parameter(torch.zeros(num_layers-1))

        for layer in range(num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears.append(nn.Linear(hidden_dim, output_dim))
                 
    def preprocess_graph_pool(self, batch_graph):
        # sum or average pooling sparse matrix (num graphs X num nodes)
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.nx_graph))

        idx = []
        val = []

        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == 'sum':
                val.extend([1] * len(graph.nx_graph) )
            else:
                val.extend([1.0/len(graph.nx_graph)] * graph.nx_graph)
            
            idx.extend([[i,j] for j in range(start_idx[i], start_idx[i+1])])
        idx = torch.LongTensor(idx).transpose(0, 1)
        val = torch.FloatTensor(val)
        graph_pool = torch.sparse.FloatTensor(idx, val, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)
    

    def neighbor_sum_avg_mask(self, batch_graph):

        start_idx = [0]
        edges_list = []
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.nx_graph))
            edges_list.append(graph.edges + start_idx[i]) # [tensor...]
        adj_idx = torch.cat(edges_list, 1)
        adj_val = torch.ones(adj_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            self_loop_edges = torch.LongTensor([range(start_idx[-1]), range(start_idx[-1])])
            self_loop_val = torch.ones(start_idx[-1])

            adj_idx = torch.cat([adj_idx, self_loop_edges], 1)
            adj_val = torch.cat([adj_val, self_loop_val], 0)

        batch_adj = torch.sparse.FloatTensor(adj_idx, adj_val, torch.Size([start_idx[-1], start_idx[-1]]))

        return batch_adj.to(self.device)

    def neighbor_maxpadding_mask(self, batch_graph):
        max_degree = max([graph.max_degree for graph in batch_graph])

        start_idx = [0]
        padded_neighbor_list = []  #( number of all neighbors,  max_degree)
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.nx_graph)) 
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                pad = [k + start_idx[i] for k in graph.neighbors[j]]
                pad.extend([-1] * (max_degree - len(pad)))

                if not self.learn_eps:
                    pad.append(j + start_idx[i])
                
                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def max_aggregate(self, h, padded_neighbor_list):
        dummy = torch.min(h, dim = 0)[0]
       
        # to ensure dummy data is min  
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep 


    def next_layer(self, h, layer, padded_neighbor_list=None, adj=None):
        # aggregate function is max 
        if self.neighbor_aggregating_type == 'max':
            aggregated = max_aggregate(h, padded_neighbor_list)
        else:
            # aggregate function is sum 
            aggregated = torch.spmm(adj, h)
            # aggregate function is average 
            if self.neighbor_aggregating_type == 'average':
                degree = torch.spmm(adj, torch.ones((adj.shape[0], 1)).to(self.device))
                aggregated /= degree
        
        if self.learn_eps:
            aggregated = aggregated + (1 + self.eps[layer])*h
        
        h = self.mlps[layer](aggregated)
        h = self.batch_norms[layer](h)

        #non-linearity
        h = F.relu(h)
        return h
            


    def forward(self, batch_graph):

        graph_pool = self.preprocess_graph_pool(batch_graph)

        if not self.neighbor_aggregating_type == 'max':
            batch_adj = self.neighbor_sum_avg_mask(batch_graph)
        else:
            padded_neighbor_list = self.neighbor_maxpadding_mask(batch_graph)

        # (len(graph1.node_labels) + len(graph1.node_labels) +...  , len(node_label_map))
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        
        hiddens = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_aggregating_type == 'max':
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            else:
                h = self.next_layer(h, layer, adj=batch_adj)

            hiddens.append(h)
        
        score_over_layer = 0

        for layer, h in enumerate(hiddens):
            pooled = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears[layer](pooled), self.final_drop_out)
        
        return score_over_layer
        

    

    
            


