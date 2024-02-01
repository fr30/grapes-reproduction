import hydra
import time
import torch
import torch.nn.functional as F

from src.data import get_dataset
from torch_geometric.logging import log
from src.models import GCN


@hydra.main(version_base=None, config_path="configs", config_name="fullbatch")
def main(cfg):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    dataset = dataset = get_dataset(cfg.dataset)
    data = dataset.data.to(device)
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=cfg.hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs

    times = []
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        loss = train()
        train_acc, val_acc, test_acc = test()
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


if __name__ == "__main__":
    main()
