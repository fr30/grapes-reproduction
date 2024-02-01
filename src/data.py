from torch_geometric.datasets import Planetoid, Reddit2, Flickr
import torch_geometric.transforms as T


def get_planetoid(name):
    dataset = Planetoid(
        root="data", name=name, split="full", transform=T.NormalizeFeatures()
    )
    return dataset


def get_reddit(_):
    dataset = Reddit2(root="data/Reddit2", transform=T.NormalizeFeatures())
    # dataset = Reddit2(f"data/Reddit2")
    # data = dataset[0]
    # data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return dataset


def get_flickr(_):
    dataset = Flickr(root="data/Flickr", transform=T.NormalizeFeatures())
    return dataset


def get_dataset(name):
    name_to_fun = {
        "cora": get_planetoid,
        "citeseer": get_planetoid,
        "pubmed": get_planetoid,
        # "reddit": get_reddit,
        "flickr": get_flickr,
    }
    return name_to_fun[name.lower()](name.lower())
