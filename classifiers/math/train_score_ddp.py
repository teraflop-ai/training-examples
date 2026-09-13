import argparse
import os
from datetime import timedelta

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from torchmetrics import MeanMetric, MeanSquaredError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"/e/data1/datasets/playground/mmlaion/shared/enrico/models/jhu-clsp/ettin-encoder-32m")
    p.add_argument("--train-files", nargs="+", default=[f"/e/data1/datasets/playground/mmlaion/shared/enrico/datasets/quality_labels/train/*.parquet"])
    p.add_argument("--val-files", nargs="+", default=[f"/e/data1/datasets/playground/mmlaion/shared/enrico/datasets/quality_labels/val/*.parquet"])
    p.add_argument("--out", default=f"/e/data1/datasets/playground/mmlaion/shared/enrico/models/ettin_quality_32m")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", "--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", "--maxlen", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--decay-ratio", type=float, default=0.2)
    p.add_argument("--kernel-repo", default="kernels-community/flash-attn2")
    p.add_argument("--kernel-revision", default="v3")
    p.add_argument("--kernel-build", default="torch-stable-abi210-cu130-aarch64-linux")
    p.add_argument("--kernels-cache", default=os.environ.get("KERNELS_CACHE"))
    p.add_argument("--attn", default="kernels-community/flash-attn2@v3")
    return p.parse_args()


def tokenize(batch, tok, max_length):
    enc = tok(batch["content"], truncation=True, max_length=max_length)
    enc["labels"] = [float(s) for s in batch["score"]]
    return enc


def to_device(batch):
    return {k: v.cuda(non_blocking=True) for k, v in batch.items()}


def train_epoch(model, loader, opt, sched, rank, epoch, args):
    model.train()
    torch.cuda.reset_peak_memory_stats()
    bar = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}", disable=rank != 0)
    for batch in bar:
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad()
        if rank == 0:
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{sched.get_last_lr()[0]:.2e}",
                max_gpu=f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mse = MeanSquaredError().cuda()
    acc = MeanMetric().cuda()
    acc3 = MeanMetric().cuda()

    for batch in tqdm(loader, desc="Eval", disable=dist.get_rank() != 0):
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)

        labels = batch["labels"]
        logits = out.logits.squeeze(-1).float()
        pred = logits.round().clamp(0, 5)

        mse.update(logits, labels)
        acc.update((pred == labels).float())
        acc3.update(((pred >= 3) == (labels >= 3)).float())

    return tuple(metric.compute().item() for metric in (mse, acc, acc3))


def main():
    args = parse_args()
    if args.attn == f"{args.kernel_repo}@{args.kernel_revision}":
        from huggingface_hub import snapshot_download

        kernel_path = snapshot_download(
            args.kernel_repo,
            repo_type="kernel",
            revision=args.kernel_revision,
            cache_dir=args.kernels_cache,
            local_files_only=True,
        )
        os.environ["LOCAL_KERNELS"] = (
            f"{args.kernel_repo}={kernel_path}/build/{args.kernel_build}"
        )

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_wsd_schedule,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(hours=1),
        device_id=torch.device("cuda", local_rank),
    )
    rank = dist.get_rank()

    tok = AutoTokenizer.from_pretrained(args.model)
    if rank != 0:
        dist.barrier()
    ds = load_dataset(
        "parquet",
        data_files={"train": args.train_files, "val": args.val_files},
        num_proc=args.num_proc,
    )
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={"tok": tok, "max_length": args.max_length},
        num_proc=args.num_proc,
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
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        attn_implementation=args.attn,
    ).cuda()
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    sched = get_wsd_schedule(
        opt,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_decay_steps=int(args.decay_ratio * total_steps),
        num_training_steps=total_steps,
        decay_type="1-sqrt",
        min_lr_ratio=0.0,
    )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, opt, sched, rank, epoch, args)
        val_mse, val_acc, val_acc3 = evaluate(model, val_loader)
        if rank == 0:
            print(
                f"epoch {epoch + 1} val_mse {val_mse:.4f} "
                f"val_acc {val_acc:.4f} val_acc@3 {val_acc3:.4f}",
                flush=True,
            )

    if rank == 0:
        model.module.save_pretrained(args.out)
        tok.save_pretrained(args.out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
