import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_wsd_schedule,
)

SHARED = "/e/data1/datasets/playground/mmlaion/shared/enrico"
MODEL = f"{SHARED}/models/jhu-clsp/ettin-encoder-68m"
EPOCHS = 2
BS = 128
LR = 3e-5
MAXLEN = 2048
NUM_LABELS = 1
NUM_PROC = 256
NUM_WORKERS = 8
ATTN = "kernels-community/flash-attn2@v3"
BASE = f"{SHARED}/datasets"
OUT = f"{BASE}/ettin_quality_68m"


def tokenize(batch, tok):
    enc = tok(batch["content"], truncation=True, max_length=MAXLEN)
    enc["labels"] = [float(s) for s in batch["score"]]
    return enc


def to_device(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def train_epoch(model, loader, opt, sched, rank, epoch):
    model.train()
    bar = tqdm(loader, desc=f"Train {epoch + 1}/{EPOCHS}", disable=rank != 0)
    for batch in bar:
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if rank == 0:
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{sched.get_last_lr()[0]:.2e}",
            )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    totals = torch.zeros(4, device="cuda")
    for batch in tqdm(loader, desc="Eval", disable=dist.get_rank() != 0):
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        labels = batch["labels"]
        pred = out.logits.squeeze(-1).float().round().clamp(0, 5)
        exact = (pred == labels).sum().item()
        bin3 = ((pred >= 3) == (labels >= 3)).sum().item()
        n = len(labels)
        totals += torch.tensor(
            [out.loss.item() * n, exact, bin3, n], device="cuda"
        )
    dist.all_reduce(totals)
    loss_sum, exact_sum, bin3_sum, count = totals.tolist()
    return loss_sum / count, exact_sum / count, bin3_sum / count


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(hours=1),
        device_id=torch.device("cuda", local_rank),
    )
    rank = dist.get_rank()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if rank != 0:
        dist.barrier()
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{BASE}/quality_labels/train/*.parquet",
            "val": f"{BASE}/quality_labels/val/*.parquet",
        },
        num_proc=NUM_PROC,
    )
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={"tok": tok},
        num_proc=NUM_PROC,
        remove_columns=["content", "score"],
    )
    if rank == 0:
        print("data ready", flush=True)
        dist.barrier()

    collate = DataCollatorWithPadding(tok)
    train_sampler = DistributedSampler(ds["train"])
    val_sampler = DistributedSampler(ds["val"], shuffle=False)
    train_loader = DataLoader(
        ds["train"],
        batch_size=BS,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=BS,
        sampler=val_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=NUM_LABELS,
        attn_implementation=ATTN,
    ).cuda()
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = EPOCHS * len(train_loader)
    sched = get_wsd_schedule(
        opt,
        num_warmup_steps=int(0.1 * total_steps),
        num_decay_steps=int(0.2 * total_steps),
        num_training_steps=total_steps,
        decay_type="1-sqrt",
        min_lr_ratio=0.0,
    )

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, opt, sched, rank, epoch)
        val_mse, val_acc, val_acc3 = evaluate(model, val_loader)
        if rank == 0:
            print(
                f"epoch {epoch + 1} val_mse {val_mse:.4f} "
                f"val_acc {val_acc:.4f} val_acc@3 {val_acc3:.4f}",
                flush=True,
            )

    if rank == 0:
        model.module.save_pretrained(OUT)
        tok.save_pretrained(OUT)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()