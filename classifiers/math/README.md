
# Training models with
```
srun -A reformo -p booster \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=72 \
  --gres=gpu:4 --time=12:00:00 --job-name=enrico \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  /e/project1/reformo/enrico/training-examples/classifiers/math/train_topic_ddp.py
```

# Uploading trained models from the CLI
```sh
unset HF_HUB_OFFLINE
hf repos create TeraflopAI/ettin-150m-math-topic --type model
```
```sh
unset HF_HUB_OFFLINE
hf upload 'TeraflopAI/ettin-150m-math-topic' /e/data1/datasets/playground/mmlaion/shared/enrico/datasets/ettin_topics_150m . --repo-type model
```
