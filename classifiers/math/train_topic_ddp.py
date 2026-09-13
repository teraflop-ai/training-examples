import argparse
import json
import os
from datetime import timedelta

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/e/data1/datasets/playground/mmlaion/shared/enrico/models/jhu-clsp/ettin-encoder-150m")
    p.add_argument(
        "--train-files",
        nargs="+",
        default=["/e/data1/datasets/playground/mmlaion/shared/enrico/datasets/topic_labels/train/*.parquet"],
    )
    p.add_argument(
        "--val-files",
        nargs="+",
        default=["/e/data1/datasets/playground/mmlaion/shared/enrico/datasets/topic_labels/val/*.parquet"],
    )
    p.add_argument("--out", default="/e/data1/datasets/playground/mmlaion/shared/enrico/models/ettin_topics_150m")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", "--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", "--maxlen", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--decay-ratio", type=float, default=0.2)
    p.add_argument("--label-column", default="labels")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--kernel-repo", default="kernels-community/flash-attn2")
    p.add_argument("--kernel-revision", default="v3")
    p.add_argument("--kernel-build", default="torch-stable-abi210-cu130-aarch64-linux")
    p.add_argument("--kernels-cache", default=os.environ.get("KERNELS_CACHE"))
    p.add_argument("--attn", default="kernels-community/flash-attn2@v3")
    return p.parse_args()


def get_topics(value):
    return (json.loads(value) if isinstance(value, str) else value)["topics"]


def tokenize(batch, tok, label2id, max_length, label_column):
    enc = tok(batch["content"], truncation=True, max_length=max_length)
    labels = []
    for value in batch[label_column]:
        target = [0.0] * len(label2id)
        for topic in get_topics(value):
            if topic not in label2id:
                raise ValueError(f"Topic absent from training vocabulary: {topic}")
            target[label2id[topic]] = 1.0
        labels.append(target)
    enc["labels"] = labels
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
def evaluate(model, loader, args):
    model.eval()
    bce = MeanMetric().cuda()
    exact = MeanMetric().cuda()
    micro_f1 = BinaryF1Score(zero_division=0).cuda()

    for batch in tqdm(loader, desc="Eval", disable=dist.get_rank() != 0):
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)

        labels = batch["labels"].int()
        pred = (out.logits.float().sigmoid() >= args.threshold).int()

        bce.update(out.loss.float(), weight=len(labels))
        exact.update((pred == labels).all(dim=1).float())
        micro_f1.update(pred.flatten(), labels.flatten())

    return tuple(metric.compute().item() for metric in (bce, exact, micro_f1))


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
    if len(ds["val"]) == 0:
        raise ValueError("Validation split is empty")

    topics = sorted({
        topic
        for value in ds["train"][args.label_column]
        for topic in get_topics(value)
    })
    if not topics:
        raise ValueError("No topics found in training data")

    label2id = {topic: i for i, topic in enumerate(topics)}
    ds = {
        split: data.map(
            tokenize,
            batched=True,
            fn_kwargs={
                "tok": tok,
                "label2id": label2id,
                "max_length": args.max_length,
                "label_column": args.label_column,
            },
            num_proc=args.num_proc,
            remove_columns=data.column_names,
        )
        for split, data in ds.items()
    }
    if rank == 0:
        print(f"data ready; {len(topics)} topics", flush=True)
        dist.barrier()

    collate = DataCollatorWithPadding(tok)
    train_sampler = DistributedSampler(ds["train"])
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
        sampler=range(rank, len(ds["val"]), dist.get_world_size()),
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(topics),
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label={i: topic for topic, i in label2id.items()},
        ignore_mismatched_sizes=True,
        attn_implementation=args.attn,
    ).cuda()
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        val_bce, val_exact, val_f1 = evaluate(model.module, val_loader, args)
        if rank == 0:
            print(
                f"epoch {epoch + 1} val_bce {val_bce:.4f} "
                f"val_exact {val_exact:.4f} val_micro_f1 {val_f1:.4f}",
                flush=True,
            )

    if rank == 0:
        model.module.save_pretrained(args.out)
        tok.save_pretrained(args.out)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()