import hydra
import time
import torch
import torch.nn.functional as F

from src.data import get_dataset
from src.models import GCN, GFlowNet2
from src.sampler import GrapesSampler
from torch_geometric.logging import log


@hydra.main(version_base=None, config_path="configs", config_name="grapes")
def main(cfg):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    dataset = get_dataset(cfg.dataset)
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=cfg.hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)
    gfnet = GFlowNet2(
        in_channels=dataset.num_features,
        hidden_channels=cfg.hidden_channels,
        k=cfg.sample_size,
        beta=cfg.beta,
    ).to(device)
    sampler = GrapesSampler(gfnet, cfg.sample_size, dataset, batch_size=cfg.batch_size)
    optimizer_gcn = torch.optim.Adam(model.parameters(), lr=cfg.lr_gcn)
    optimizer_gfnet = torch.optim.Adam(gfnet.parameters(), lr=cfg.lr_gfnet)

    torch.autograd.set_detect_anomaly(True)

    def train():
        model.train()
        running_cl_loss = 0.0
        running_tb_loss = 0.0
        for local_nodes, x, adj, y, f_probs in sampler.train_iter():
            local_nodes = local_nodes.to(device)
            x = x.to(device)
            adj = [a.to(device) for a in adj]
            y = y.to(device)

            optimizer_gcn.zero_grad()
            out = model(x, adj)
            cl_loss = F.cross_entropy(out[local_nodes], y)
            cl_loss.backward()
            optimizer_gcn.step()

            optimizer_gfnet.zero_grad()
            reward = cl_loss.detach()
            logz = torch.log(gfnet.z)
            tb_loss = (logz + f_probs.sum() + cfg.alpha * reward) ** 2

            tb_loss.backward()
            optimizer_gfnet.step()
            gfnet.update_z(reward)

            running_cl_loss += float(cl_loss)
            running_tb_loss += float(tb_loss)
        return (
            running_cl_loss * cfg.batch_size / sampler.num_train,
            running_tb_loss * cfg.batch_size / sampler.num_train,
        )

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
            for local_nodes, x, adj, y, _ in iter:
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
        cl_loss, tb_loss = train()
        train_acc, val_acc, test_acc = test()
        log(
            Epoch=epoch,
            ClLoss=cl_loss,
            TbLoss=f"{tb_loss / 1000:.2f}e3",
            Train=train_acc,
            Val=val_acc,
            Test=test_acc,
        )
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


if __name__ == "__main__":
    main()
