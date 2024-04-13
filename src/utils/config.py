import os

import yaml

# Automatically set the base directory to the root of the repository
if os.path.exists("/scratch/xiangyj/moe"):
    BASEDIR = "/scratch/xiangyj/moe"
    ckpt_dir = "/scratch/xiangyj/moe/data"
    result_dir_base = "/scratch/xiangyj/moe/samsung_profiling_gpu_budget"
else:
    BASEDIR = os.path.abspath(os.path.join(__file__, "../../.."))
    ckpt_dir = os.path.join(os.path.dirname(__file__), "../../data")
    result_dir_base = os.path.join(os.path.dirname(__file__), "../../results")

_default_config_for_accuracy = {
    "defaults": ["_self_", "task: pt", "local_env: default"],
    "mode": "pt",
    "device": "gpu",
    "precision": "bf16",
    "eval_only": False,
    "predict_only": False,
    "seed": 2137,
    "model": {
        "klass": "local_t5",
        "name": "google/t5-v1_1-base",
        "overwrite": {"dropout_rate": 0.0},
        "add_config": {"is_bf16": False},
        "checkpoint_path": "",
        "random_init": True,
        "compile": False,
    },
    "data": {"input_length": 512, "mlm_probability": 0.15, "mean_noise_span_length": 3.0, "num_workers": 2},
    "optim": {
        "name": "adamwscale",
        "base_lr": 2e-2,
        "batch_size": 64,
        "total_steps": 65536,
        "epochs": -1,
        "warmup_steps": 10000,
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "grad_acc": 1,
        "final_cosine": 1e-5,
    },
    "eval": {"every_steps": 100000, "steps": 500},
    "checkpoint": {"every_steps": 100000},
    "logging": {
        "neptune": False,
        "neptune_creds": {"project": None, "api_token": None, "tags": ""},
        "every_steps": 100,
        "grad_l2": True,
        "weights_l2": True,
    },
    "hydra": {"job": {"chdir": True}},
}


class Args:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to Config objects
                setattr(self, key, Args(value))
            else:
                setattr(self, key, value)


def load_config(config_path=None):
    if config_path is None or not os.path.exists(config_path):
        config_dict = _default_config_for_accuracy
    else:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
    return Args(config_dict)
