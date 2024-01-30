import hydra
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from src.models import GCN
from src.sampler import UniformSampler


@hydra.main(version_base=None, config_path="configs", config_name="uniform")
def main(cfg):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    dataset = Planetoid(
        root="data", name="Cora", split="full", transform=T.NormalizeFeatures()
    )
    sampler = UniformSampler(cfg.sample_size, dataset, batch_size=cfg.batch_size)
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=cfg.hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def train():
        model.train()
        running_loss = 0.0
        running_acc = 0
        for local_nodes, x, adj, y in sampler.train_iter():
            local_nodes = local_nodes.to(device)
            x = x.to(device)
            adj = [a.to(device) for a in adj]
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x, adj)
            loss = F.cross_entropy(out[local_nodes], y)
            loss.backward()
            optimizer.step()

            running_acc += int((out[local_nodes].argmax(dim=-1) == y).sum())
            running_loss += float(loss)
        return running_loss * cfg.batch_size / sampler.num_train

    @torch.no_grad()
    def test_full():
        data = dataset.data.to(device)
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs

    @torch.no_grad()
    def test_minibatch():
        model.eval()
        accs = []
        data_iters = [
            (sampler.train_iter(), sampler.num_train),
            (sampler.val_iter(), sampler.num_val),
            (sampler.test_iter(), sampler.num_test),
        ]
        for iter, iter_len in data_iters:
            correct = 0
            for local_nodes, x, adj, y in iter:
                local_nodes = local_nodes.to(device)
                x = x.to(device)
                adj = [a.to(device) for a in adj]
                y = y.to(device)

                pred = model(x, adj)[local_nodes].argmax(dim=-1)
                correct += int((pred == y).sum())
            accs.append(correct / iter_len)

        return accs

    times = []
    test = test_full if cfg.test_full else test_minibatch
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        loss = train()
        train_acc, val_acc, test_acc = test()
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


if __name__ == "__main__":
    main()
