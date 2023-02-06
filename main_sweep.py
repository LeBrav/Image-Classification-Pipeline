import wandb
import argparse
import sys
from utils.io import load_yaml_config
from main import main
import functools


def main_sweep(sweep_path, sweep_count, wandb_pname, inference, distilation):
    cfg_sweep = load_yaml_config(sweep_path)

    sweep_id = wandb.sweep(cfg_sweep, project=wandb_pname, entity="bipulantes")
    wandb_main = functools.partial(main, wandb_pname, inference, distilation)
    wandb.agent(sweep_id, wandb_main, count=sweep_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="sweeps/sweep_resnet.yaml")
    parser.add_argument("--n_runs", default=25, type=int)
    parser.add_argument("--wandb", default="M3-Week5") #If default=None no wandb will be used
    parser.add_argument("--inference", default=False, type=bool)
    parser.add_argument("--distilation", default=False, type=bool)

    args = parser.parse_args(sys.argv[1:])
    sweep = args.sweep
    n_runs = args.n_runs
    wandb_name = args.wandb
    inference = args.inference
    distilation = args.distilation

    main_sweep(sweep, n_runs, wandb_name, inference, distilation)
