import wandb


def init_wandb_proj(cfg, project_name):
    run = None
    if cfg is None:
        # Sweep

        run = wandb.init(config=cfg, entity="bipulantes")

        cfg = wandb.config
        cfg = nested_dict(cfg)
        sweep_run_name = f"default_name"
        run.name = sweep_run_name

    elif project_name is not None:
        run = wandb.init(project=project_name, config=cfg, entity="bipulantes")

    return cfg, run


def nested_dict(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict
