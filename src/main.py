import argparse
import os

import torch
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import get_text_cls_loader, get_mlm_dataloaders
from model import SwitchTransformersClassificationModel
from utils import config, get_module_memory_usage, monitor_gputil
from utils.hooks import inputHook, routerHook

BASEDIR = config.BASEDIR
SPLIT = "validation"


def main():
    options = argparse.ArgumentParser()
    options.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "mrpc", "rte", "multirc", "c4_en"],
        help="Run experiments on which dataset",
    )
    options.add_argument(
        "--model",
        type=str,
        default="switch-base-128",
        choices=[
            "switch-large-128",
            "switch-xxl-128",
            "switch-c-2048",
            "switch-base-128",
            "switch-base-256",
            "switch-base-64",
            "switch-base-32",
            "switch-base-16",
            "switch-base-8",
        ],
        help="Run experiments on which model",
    )
    options.add_argument("--hf_offload", action="store_true", help="Use huggingface offloading")
    options.add_argument("--sharding", action="store_true", help="Store the Sharded Weight in this run")
    options.add_argument("--save_hash", type=bool, default=True, help="Whether to save the true hash table")
    args = options.parse_args()

    DATASET = args.dataset
    MODEL = args.model
    exp_dir = f"{BASEDIR}/data/{DATASET}/{MODEL}/"

    try:
        os.makedirs(exp_dir)
        print(f"Successfully created the directory {exp_dir}")
    except FileExistsError:
        print(f"Directory {exp_dir} already exists")
    except OSError:
        print(f"Creation of the directory {exp_dir} failed")

    # Initialize custom model
    model = SwitchTransformersClassificationModel(2, MODEL, BASEDIR, meta=args.hf_offload)

    # Load the dataset
    if DATASET != "c4_en":
        eval_loader, _, prec_func = get_text_cls_loader(DATASET, MODEL, f"{BASEDIR}/tmp/", batch_size=64)
    else:
        args_ds = config.load_config(os.path.join(BASEDIR, "src", "./default.yaml"))
        args_ds.tmp_dir = os.path.join(BASEDIR, "tmp")
        args_ds.dataset = DATASET
        tokenizer = AutoTokenizer.from_pretrained(f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/")

        _, eval_loader = get_mlm_dataloaders(tokenizer, model.config, args_ds)
        prec_func = lambda inputs: {k: v.to("cuda") for k, v in inputs.items()}

    print("=" * 86)
    print(f"{get_module_memory_usage(model)/ 1024**3:.6f} GB")
    moe_param = 0
    for name, module in model.switch_transformers.encoder.block.named_children():
        if int(name) % 2 == 1:
            moe_param += get_module_memory_usage(module) / 1024**3
        print(f"[{name}]: {get_module_memory_usage(module)/ 1024**3:.6f} GB")
    print(f"[Experts] ", moe_param)

    sharded_path = os.path.join(BASEDIR, f"data/shard_models/{MODEL}")

    if args.sharding:
        acc = Accelerator()
        acc.save_model(model, sharded_path, max_shard_size="2GB")
        print(f"Sharded Weights Saved to {sharded_path}")
        exit(0)

    if args.hf_offload:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=sharded_path,
            max_memory={0: "18GB", "cpu": "1024GB"},
            offload_folder=f"{BASEDIR}/tmp/",
            device_map="auto",
        )
    else:
        model.to("cuda")

    if args.save_hash:
        router_hook = routerHook(model, save_dir=exp_dir)
        if DATASET != "c4_en":
            input_hook = inputHook(model, save_dir=exp_dir)

    target_device = "cuda" if not args.hf_offload else list(model.hf_device_map.values())[0]
    with monitor_gputil():
        # Evaluate the model
        for step, inputs in tqdm(enumerate(eval_loader)):
            inputs = prec_func(inputs)
            inputs.update({"attention_mask":None})
            with torch.no_grad():
                outputs = model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )

    if args.save_hash:
        torch.save(
            router_hook.activation_dict,
            f"{BASEDIR}/data/{DATASET}/{MODEL}/activation_{SPLIT}_large-{router_hook.name_ptr}.pt",
        )
        

        if DATASET != "c4_en":
            torch.save(
                input_hook.data_dict, f"{BASEDIR}/data/{DATASET}/{MODEL}/data_{SPLIT}_large-{input_hook.name_ptr}.pt"
            )


if __name__ == "__main__":
    main()
