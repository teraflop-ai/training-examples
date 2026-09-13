```sh
srun -A reformo -p booster \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=72 \
  --gres=gpu:4 --time=12:00:00 --job-name=ddp \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  /e/project1/reformo/enrico/training-examples/classifiers/train_ddp.py
```

# Enter an interactive node on Jupiter
```sh
srun -A reformo -p booster \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=72 \
  --gres=gpu:4 --time=12:00:00 --job-name=ddp \
  --pty bash
```