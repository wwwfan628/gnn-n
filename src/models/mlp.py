import torch.nn.functional as F
import torch.nn as nn
import dgl

class MLP_4node(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, out_feats):
        super(MLP_4node, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        if self.num_layers != 1:
            # input layer
            self.fc_layers.append(nn.Linear(in_feats, h_feats))
            # intermediate layers
            for l in range(1, self.num_layers-1):
                self.fc_layers.append(nn.Linear(h_feats, h_feats))
            # output layer
            self.fc_layers.append(nn.Linear(h_feats, out_feats))
        else:
            self.fc_layers.append(nn.Linear(in_feats, out_feats))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers-1):
            h = F.relu(self.fc_layers[l](h))
        h = self.fc_layers[self.num_layers-1](h)
        return h



class MLP_4graph(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, out_feats):
        super(MLP_4graph, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        if self.num_layers != 1:
            # input layer
            self.fc_layers.append(nn.Linear(in_feats, h_feats))
            # intermediate layers
            for l in range(1, self.num_layers-1):
                self.fc_layers.append(nn.Linear(h_feats, h_feats))
            # output layer
            self.fc_layers.append(nn.Linear(h_feats, out_feats))
        else:
            self.fc_layers.append(nn.Linear(in_feats, out_feats))

    def forward(self, graph, inputs):
        h = inputs
        for l in range(self.num_layers-1):
            h = F.relu(self.fc_layers[l](h))

        graph.ndata['h'] = h
        # read-out function
        graph_repr = dgl.mean_nodes(graph, 'h')
        h = self.fc_layers[self.num_layers-1](graph_repr)
        return h