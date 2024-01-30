import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_conv_layers=2, dropout=0.0
    ):
        super().__init__()
        conv_layers = []
        conv_layers += [GCNConv(in_channels, hidden_channels)]
        for _ in range(1, num_conv_layers - 1):
            conv_layers += [GCNConv(hidden_channels, hidden_channels)]
        conv_layers += [GCNConv(hidden_channels, out_channels)]

        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

    def forward(self, x, adj):
        if type(adj) is not list:
            adj = [adj for _ in range(self.num_conv_layers)]
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, adj[i])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, adj[-1])
        return x
