import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"

from huggingface_hub import snapshot_download

kernel_path = snapshot_download(
    "kernels-community/flash-attn2",
    repo_type="kernel",
    revision="v3",
    cache_dir=os.environ.get("KERNELS_CACHE"),
    local_files_only=True,
)
os.environ["LOCAL_KERNELS"] = (
    f"kernels-community/flash-attn2={kernel_path}/build/"
    "torch-stable-abi210-cu130-aarch64-linux"
)

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
MODEL = f"{SHARED}/models/jhu-clsp/ettin-encoder-150m"
EPOCHS = 3
BS = 256
LR = 3e-5
MAXLEN = 2048
NUM_PROC = 256
NUM_WORKERS = 8
ATTN = "kernels-community/flash-attn2@v3"
BASE = f"{SHARED}/datasets"
DATA = f"{BASE}/topic_labels"
OUT = f"{BASE}/ettin_topics_150m"
LABEL_COLUMN = "topics"
THRESHOLD = 0.5


def get_topics(value):
    return (json.loads(value) if isinstance(value, str) else value)["topics"]


def tokenize(batch, tok, label2id):
    enc = tok(batch["content"], truncation=True, max_length=MAXLEN)
    labels = []
    for value in batch[LABEL_COLUMN]:
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


def train_epoch(model, loader, opt, sched, rank, epoch):
    model.train()
    torch.cuda.reset_peak_memory_stats()
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
                max_gpu=f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
            )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    totals = torch.zeros(6, device="cuda", dtype=torch.float64)
    for batch in tqdm(loader, desc="Eval", disable=dist.get_rank() != 0):
        batch = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        labels = batch["labels"].bool()
        pred = out.logits.float().sigmoid() >= THRESHOLD
        totals[0] += out.loss.double() * len(labels)
        totals[1] += (pred == labels).all(dim=1).sum()
        totals[2] += (pred & labels).sum()
        totals[3] += (pred & ~labels).sum()
        totals[4] += (~pred & labels).sum()
        totals[5] += len(labels)
    dist.all_reduce(totals)
    loss_sum, exact, tp, fp, fn, count = totals.tolist()
    if count == 0:
        raise ValueError("Validation split is empty")
    micro_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return loss_sum / count, exact / count, micro_f1


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
            "train": f"{DATA}/train/*.parquet",
            "val": f"{DATA}/val/*.parquet",
        },
        num_proc=NUM_PROC,
    )
    topics = sorted({
        topic
        for value in ds["train"][LABEL_COLUMN]
        for topic in get_topics(value)
    })
    if not topics:
        raise ValueError("No topics found in training data")
    label2id = {topic: i for i, topic in enumerate(topics)}
    ds = {
        split: data.map(
            tokenize,
            batched=True,
            fn_kwargs={"tok": tok, "label2id": label2id},
            num_proc=NUM_PROC,
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
        batch_size=BS,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=BS,
        sampler=range(rank, len(ds["val"]), dist.get_world_size()),
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=len(topics),
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label={i: topic for topic, i in label2id.items()},
        ignore_mismatched_sizes=True,
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
        val_bce, val_exact, val_f1 = evaluate(model.module, val_loader)
        if rank == 0:
            print(
                f"epoch {epoch + 1} val_bce {val_bce:.4f} "
                f"val_exact {val_exact:.4f} val_micro_f1 {val_f1:.4f}",
                flush=True,
            )

    if rank == 0:
        model.module.save_pretrained(OUT)
        tok.save_pretrained(OUT)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()