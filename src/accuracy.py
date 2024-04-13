import argparse
import os

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

import utils.config as config
from dataset import get_mlm_dataloaders
from model import SwitchTransformersForConditionalGenerationOffload
from utils import load_oracle_hash

BASEDIR = config.BASEDIR


def accuracy(model, tokenizer, device, topk=1, offload=False, model_name=None, dataset_split=None, dataset=None):
    args = config.load_config(os.path.join(BASEDIR, "src", "./default.yaml"))
    args.tmp_dir = os.path.join(BASEDIR, "tmp")
    args.dataset = dataset

    if offload:
        hash_table = load_oracle_hash(
            f"{BASEDIR}/data/{dataset}/{model_name}", ratio=None, topk=topk
        )
    _, test_dataloader = get_mlm_dataloaders(tokenizer, model.config, args)
    total_loss = 0
    total_n_batches = 0

    for step, batch in enumerate(test_dataloader, start=1):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            local_hash_table = (
                [(hash_table, step, key, topk) for key in list(hash_table.keys())[::-1]] if offload else None
            )
            if offload:
                outputs = model(hash_table=local_hash_table, output_router_logits=False, **batch)
            else:
                outputs = model(**batch)
            total_loss += outputs.loss
            total_n_batches += 1
            print(f">>>[Step#{step}]", outputs.loss.item(), torch.exp(outputs.loss).item())
            break

    ppl = torch.exp(total_loss / total_n_batches).item()
    return ppl, (total_loss / total_n_batches).item()


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument(
        "--model",
        type=str,
        choices=[
            "switch-base-128",
            "switch-base-256",
            "switch-base-64",
            "switch-base-32",
            "switch-base-16",
            "switch-base-8",
        ],
        help="Run experiments on which model",
    )
    options.add_argument("--offload", action="store_true", help="Use Offload MoE")
    options.add_argument(
        "--dataset",
        type=str,
        choices=["sst2", "mrpc", "rte", "multirc", "c4_en"],
        help="Run experiments on which dataset",
    )
    options.add_argument("--topk", type=int, default=1, help="Top k experts to use")

    args = options.parse_args()
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model}", cache_dir=f"{BASEDIR}/tmp/")
    if args.offload:
        model = SwitchTransformersForConditionalGenerationOffload.from_pretrained(
            f"google/{args.model}", cache_dir=f"{BASEDIR}/tmp/"
        )
    else:
        model = SwitchTransformersForConditionalGeneration.from_pretrained(
            f"google/{args.model}", cache_dir=f"{BASEDIR}/tmp/"
        )
    model.config.decoder_start_token_id = tokenizer.pad_token_id

    model = model.to(device)

    ppl, avg_loss = accuracy(
        model,
        tokenizer,
        device,
        topk=args.topk,
        model_name=args.model,
        dataset_split="test",
        dataset=args.dataset,
        offload=args.offload,
    )
    print(">>>ppl: ", ppl)