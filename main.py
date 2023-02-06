from utils.io import load_yaml_config
from model import *
from dataset import *
from utils.wandb import *
from inference import *
import argparse
import sys
from keras.models import load_model
tf.compat.v1.disable_eager_execution()


def main(project_name, inference, distilation, config_path=None):
    if config_path is not None:
        cfg = load_yaml_config(config_path)
    else:
        cfg = None

    cfg, run = init_wandb_proj(cfg, project_name)
    train_datagen, val_datagen = get_datagen(cfg)
    model, total_params = get_model(cfg, distilation)

    if inference:
        if "model_path" in cfg["inference_model"]:
            model = load_model(cfg["inference_model"]["model_path"])
        model.load_weights(cfg["inference_model"]["weights_path"])
        func_inference_CAM(model, val_datagen.next(), cfg)
    else:
        train(cfg, model, total_params, train_datagen, val_datagen, project_name, distilation)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_Resnet50.yaml")
    parser.add_argument("--wandb", default="M3-Week5") #If default=None no wandb will be used
    parser.add_argument("--inference", default=False, type=bool)
    parser.add_argument("--distilation", default=False, type=bool)
    args = parser.parse_args(sys.argv[1:])
    config = args.config
    wandb = args.wandb
    inference = args.inference
    distilation = args.distilation

    main(wandb, inference, distilation, config)
