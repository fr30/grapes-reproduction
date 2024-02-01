import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_conv_layers=2, dropout=0.0
    ):
        super().__init__()
        self.out_channels = out_channels
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
        # else:
        #     print(f"{self.out_channels}, {x.shape}")
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, adj[i])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, adj[-1])
        return x


class GFlowNet1(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_conv_layers=2,
        k=16,
        dropout=0.0,
    ):
        super().__init__()
        self.k = k
        self.gcn = GCN(
            in_channels + num_conv_layers, hidden_channels, 1, num_conv_layers, dropout
        )
        self.logz = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = torch.sigmoid(x)
        return x.squeeze()

    @property
    def device(self):
        return next(self.parameters()).device


class GFlowNet2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_conv_layers=2,
        k=16,
        dropout=0.0,
        beta=0.1,
    ):
        super().__init__()
        self.k = k
        self.gcn = GCN(
            in_channels + num_conv_layers, hidden_channels, 1, num_conv_layers, dropout
        )
        self.z = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.beta = torch.tensor(beta)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = torch.sigmoid(x)
        return x.squeeze()

    def update_z(self, reward):
        self.z.copy_(self.beta * self.z + (1 - self.beta) * reward)

    @property
    def device(self):
        return next(self.parameters()).device
