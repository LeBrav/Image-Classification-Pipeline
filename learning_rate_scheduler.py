import wandb
import numpy as np


def warmup_cosine_decay(epoch, lr, cfg, steps_per_epoch):
    warmup_steps = cfg["optimizer"]["warmup_steps"]
    total_steps = steps_per_epoch//cfg["dataset"]["batch_size"]
    cosine_decay = cfg["optimizer"]["cosine_decay"]
    if epoch < warmup_steps:
        lr = lr * (epoch + 1) / warmup_steps
    else:
        lr = cosine_decay * lr * (1 + np.cos(np.pi * (epoch - warmup_steps) / (total_steps - warmup_steps)))
    wandb.log({"lr": lr})
    return float(lr)
