import torch
from abc import abstractmethod


class GenericSampler:
    def __init__(
        self,
        dataset,
        batch_size=1,
        num_nbr_layers=2,
    ):
        data = dataset.data
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.val_mask = data.val_mask
        self.batch_size = batch_size
        self.node_features = data.x
        self.labels = data.y
        self.num_nodes = data.num_nodes
        self.num_train = data.train_mask.sum().item()
        self.num_val = data.val_mask.sum().item()
        self.num_test = data.test_mask.sum().item()
        self.num_nbr_layers = num_nbr_layers
        self.adj = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.shape[1])
        )

    def train_iter(self):
        return self._get_data_iter(self.train_mask.nonzero().view(-1))

    def test_iter(self):
        return self._get_data_iter(self.test_mask.nonzero().view(-1))

    def val_iter(self):
        return self._get_data_iter(self.val_mask.nonzero().view(-1))

    def _get_data_iter(self, all_nodes):
        node_perm = torch.randperm(all_nodes.shape[0])
        shuffled_nodes = all_nodes[node_perm]
        nodes_iter = iter(torch.split(shuffled_nodes, self.batch_size))

        def data_gen():
            for nodes in nodes_iter:
                nodes_layerwise, adj_layerwise = self._get_neighbourhoods(nodes)
                final_nodes = nodes_layerwise[-1].long()
                features = self.node_features[final_nodes]

                # Map the nodes to their indices in the final_nodes tensor
                scaled_nodes = map_elem_to_id(torch.cat([nodes, final_nodes]))
                scaled_nodes = scaled_nodes[: nodes.shape[0]]

                yield scaled_nodes, features, adj_layerwise, self.labels[nodes]

        return data_gen()

    @abstractmethod
    def _get_neighbourhoods(self, start_nodes):
        pass


# Sampler that returns full neighbourhood
class IdentitySampler(GenericSampler):
    def _get_neighbourhoods(self, start_nodes):
        nodes_layerwise = []
        adj_layerwise = []
        nodes = start_nodes.float()

        for _ in range(self.num_nbr_layers):
            nodes_layerwise.append(nodes)
            next_nodes, scaled_next_adj = self._get_single_neighbourhood(nodes)

            nodes_mask = torch.zeros(self.num_nodes)
            nodes_mask[next_nodes.long()] = 1
            nodes_mask[nodes.long()] = 1
            nodes = nodes_mask.nonzero().reshape(-1).float()

            adj_layerwise.append(scaled_next_adj)

        nodes_layerwise.append(nodes)

        return nodes_layerwise, adj_layerwise

    # Function that creates neighourhood from given start nodes.
    # It returns nodes from the next layer as well as the scaled adjacency matrix
    def _get_single_neighbourhood(self, nodes):
        onehot_nodes = get_onehot_sparse(nodes, self.adj.shape)
        sliced_adj = onehot_nodes @ self.adj

        next_nodes_mask = torch.zeros(self.num_nodes)
        next_nodes_mask[sliced_adj.indices().view(-1)] = 1
        next_nodes = next_nodes_mask.nonzero().reshape(-1).float()

        scaled_adj = map_elem_to_id(sliced_adj.indices())

        return next_nodes, scaled_adj


# Sample that uniformly samples nodes from previous neighbourhoods
class UniformSampler(GenericSampler):
    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _get_neighbourhoods(self, start_nodes):
        nodes_layerwise = []
        adj_layerwise = []
        nodes = start_nodes.float()

        for _ in range(self.num_nbr_layers):
            nodes_layerwise.append(nodes)
            next_nodes, scaled_next_adj = self._get_single_neighbourhood(nodes)

            nodes_mask = torch.zeros(self.num_nodes)
            nodes_mask[next_nodes.long()] = 1
            nodes_mask[nodes.long()] = 1
            nodes = nodes_mask.nonzero().reshape(-1).float()

            adj_layerwise.append(scaled_next_adj)

        nodes_layerwise.append(nodes)
        return nodes_layerwise, adj_layerwise

    # Function that creates neighourhood from given start nodes.
    # It returns nodes from the next layer as well as the scaled adjacency matrix
    def _get_single_neighbourhood(self, nodes):
        onehot_nodes = get_onehot_sparse(nodes, self.adj.shape)
        left_sliced_adj = onehot_nodes @ self.adj
        next_nodes = left_sliced_adj.indices().float().reshape(-1)

        # Get nodes from the next layer, excluding nodes from current layer
        nodes_mask = torch.zeros(self.num_nodes).bool()
        nodes_mask[nodes.long()] = 1
        next_nodes_mask = torch.zeros(self.num_nodes).bool()
        next_nodes_mask[next_nodes.long()] = 1

        next_nodes_mask = next_nodes_mask & ~nodes_mask
        next_layer_nodes = next_nodes_mask.nonzero().reshape(-1).float()

        # Uniformly sample which nodes to consider in the next layer
        sampled_nodes = self._sample_tensor(next_layer_nodes)
        onehot_next = get_onehot_sparse(sampled_nodes, self.adj.shape)
        sliced_adj = left_sliced_adj @ onehot_next
        scaled_adj = map_elem_to_id(sliced_adj.indices())

        return sampled_nodes, scaled_adj

    def _sample_tensor(self, x):
        perm = torch.randperm(x.size(0))
        idx = perm[: self.k]
        return x[idx]


class GrapesSampler(GenericSampler):
    def __init__(self, gflownet, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.gflownet = gflownet

    def _get_data_iter(self, all_nodes):
        node_perm = torch.randperm(all_nodes.shape[0])
        shuffled_nodes = all_nodes[node_perm]
        nodes_iter = iter(torch.split(shuffled_nodes, self.batch_size))

        def data_gen():
            for nodes in nodes_iter:
                nodes_layerwise, adj_layerwise, f_probs = self._get_neighbourhoods(
                    nodes
                )
                final_nodes = nodes_layerwise[-1].long()
                features = self.node_features[final_nodes]

                # Map the nodes to their indices in the final_nodes tensor
                scaled_nodes = map_elem_to_id(torch.cat([nodes, final_nodes]))
                scaled_nodes = scaled_nodes[: nodes.shape[0]]

                yield scaled_nodes, features, adj_layerwise, self.labels[nodes], f_probs

        return data_gen()

    # In this case get_neighbourhoods also returns forward probabilities
    # needed for GFlowNet TB objective
    def _get_neighbourhoods(self, start_nodes):
        nodes_layerwise = []
        adj_layerwise = []
        f_probs = torch.zeros(self.num_nbr_layers).to(self.gflownet.device)
        nodes = start_nodes.float()

        for hop in range(self.num_nbr_layers):
            nodes_layerwise.append(nodes)
            next_nodes, scaled_next_adj, f_prob = self._get_single_neighbourhood(
                hop, nodes
            )

            nodes_mask = torch.zeros(self.num_nodes)
            nodes_mask[next_nodes.long()] = 1
            nodes_mask[nodes.long()] = 1
            nodes = nodes_mask.nonzero().reshape(-1).float()

            adj_layerwise.append(scaled_next_adj)
            f_probs[hop] = f_prob

        nodes_layerwise.append(nodes)
        return nodes_layerwise, adj_layerwise, f_probs

    # Function that creates neighourhood from given start nodes.
    # It returns nodes from the next layer as well as the scaled adjacency matrix
    def _get_single_neighbourhood(self, hop, nodes):
        onehot_nodes = get_onehot_sparse(nodes, self.adj.shape)
        left_sliced_adj = onehot_nodes @ self.adj
        next_nodes = left_sliced_adj.indices().reshape(-1)

        all_nodes_mask = torch.zeros(self.num_nodes).bool()
        all_nodes_mask[next_nodes] = 1
        all_nodes_mask[nodes.long()] = 1
        node_map = (all_nodes_mask.cumsum(dim=0) - 1).long()
        # rev_node_map = all_nodes_mask.nonzero().reshape(-1)

        # Get nodes from the next layer, excluding nodes from current layer
        nodes_mask = torch.zeros(self.num_nodes).bool()
        nodes_mask[nodes.long()] = 1
        next_layer_nodes_mask = torch.zeros(self.num_nodes).bool()
        next_layer_nodes_mask[next_nodes] = 1
        next_layer_nodes_mask = next_layer_nodes_mask & ~nodes_mask

        # Sample which nodes to consider in the next layer using GFlowNet
        device = self.gflownet.device
        scaled_left_adj = map_elem_to_id(left_sliced_adj.indices()).to(device)
        all_node_features = self.node_features[all_nodes_mask]

        # Attach indicator features
        indicator_features = torch.zeros(
            all_node_features.shape[0], self.num_nbr_layers
        )
        indicator_features[:, hop] = 1
        all_node_features = torch.cat(
            [all_node_features, indicator_features], dim=1
        ).to(device)

        next_layer_nodes = next_layer_nodes_mask.nonzero().reshape(-1)
        scaled_next_layer_nodes = node_map[next_layer_nodes]
        # Sample probs from GFlowNet for all nodes considered
        all_probs = self.gflownet(all_node_features, scaled_left_adj)
        next_layer_nodes_probs = all_probs[scaled_next_layer_nodes]

        # Sample nodes from the probabilities
        scaled_sampled_nodes_mask, f_prob = self.sample_k_nodes(next_layer_nodes_probs)
        sampled_nodes = next_layer_nodes[scaled_sampled_nodes_mask]

        onehot_sampled = get_onehot_sparse(sampled_nodes, self.adj.shape).float()
        sliced_adj = left_sliced_adj @ onehot_sampled
        scaled_adj = map_elem_to_id(sliced_adj.indices())

        return sampled_nodes, scaled_adj, f_prob

    def sample_k_nodes(self, probs):
        logits = torch.log(probs)
        neg_logits = torch.log(1 - probs)
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        nodes = torch.topk(logits + gumbel, self.k).indices
        node_mask = torch.zeros_like(logits).bool()
        node_mask[nodes] = 1
        f_prob = torch.sum(logits[node_mask]) + torch.sum(neg_logits[~node_mask])
        return node_mask.cpu(), f_prob


# Function that maps the elements of the tensor to their indices in the uniq() tensor
# Example: [1, 2, 3, 1, 2] -> [0, 1, 2, 0, 1]
def map_elem_to_id(keys):
    num_elems = keys.max() + 1
    map = torch.zeros(num_elems, dtype=torch.long)
    map[keys] = 1
    map = map.cumsum(dim=0) - 1
    return map[keys]


def get_onehot_sparse(elems, shape):
    onehot_i = elems.repeat((2, 1))
    onehot_v = torch.ones_like(elems)
    onehot_sparse = torch.sparse_coo_tensor(onehot_i, onehot_v, shape)
    return onehot_sparse
