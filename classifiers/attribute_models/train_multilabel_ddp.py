import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm


class MultilabelClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.probe = nn.Linear(in_dim, num_labels)

    def forward(self, embeddings):
        return self.probe(embeddings)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--val-files", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("models/multilabel"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prepare_data(args):
    ds = load_dataset(
        "parquet", data_files={"train": args.train_files, "val": args.val_files}
    ).select_columns(["embeddings", "labels"])
    if not len(ds["train"]) or not len(ds["val"]):
        raise ValueError("Training and validation splits must be nonempty")
    names = sorted({label for labels in ds["train"]["labels"] for label in labels})
    if not names:
        raise ValueError("No training labels found")
    label2id = {name: i for i, name in enumerate(names)}
    if {label for labels in ds["val"]["labels"] for label in labels} - label2id.keys():
        raise ValueError("Validation contains labels absent from training")
    ds = ds.map(
        lambda row: {"labels": [float(label in row["labels"]) for label in names]}
    )
    return ds.with_format("torch"), label2id


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    bar = tqdm(loader, desc=f"Train {epoch + 1}", disable=dist.get_rank() != 0)
    for batch in bar:
        x = batch["embeddings"].to(device, dtype=torch.float32)
        y = batch["labels"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_metric = MeanMetric().to(device)
    exact = MeanMetric().to(device)
    micro_f1 = BinaryF1Score(zero_division=0).to(device)
    for batch in loader:
        x = batch["embeddings"].to(device, dtype=torch.float32)
        y = batch["labels"].to(device, dtype=torch.float32)
        logits = model(x)
        pred, target = (logits >= 0).int(), y.int()
        loss_metric.update(
            F.binary_cross_entropy_with_logits(logits, y), weight=len(y)
        )
        exact.update((pred == target).all(dim=-1).float())
        micro_f1.update(pred.flatten(), target.flatten())
    return tuple(metric.compute().item() for metric in (loss_metric, exact, micro_f1))


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(hours=1))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(args.seed)
        ds, label2id = prepare_data(args)
        sampler = DistributedSampler(ds["train"], seed=args.seed)
        train_loader = DataLoader(
            ds["train"], batch_size=args.batch_size, sampler=sampler
        )
        val_loader = DataLoader(
            ds["val"],
            batch_size=args.batch_size,
            sampler=range(rank, len(ds["val"]), world),
        )
        in_dim = len(ds["train"][0]["embeddings"])
        model = DDP(
            MultilabelClassifier(in_dim, len(label2id)).to(device),
            device_ids=[local_rank] if cuda else None,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            train_epoch(model, train_loader, optimizer, device, epoch)
            loss, exact, f1 = evaluate(model.module, val_loader, device)
            if rank == 0:
                print(
                    f"epoch {epoch + 1} val_loss={loss:.4f} "
                    f"val_exact={exact:.4f} val_micro_f1={f1:.4f}",
                    flush=True,
                )
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.module.state_dict(),
                    "in_dim": in_dim,
                    "out_dim": len(label2id),
                    "label2id": label2id,
                },
                args.out / "model.pt",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()