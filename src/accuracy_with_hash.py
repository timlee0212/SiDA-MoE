import argparse
import logging
import os

import torch
from transformers import AutoTokenizer

from dataset import get_mlm_dataloaders
from model import (
    SimpleLSTMClassifier,
    SwitchTransformersForConditionalGenerationOffload,
)
from utils import SODAManager, config, move_to_device

BASEDIR = config.BASEDIR


# Derived class for use in the conditional generation model
class SODAManagerCond(SODAManager):
    def __init__(self, model, batched_loader, n_experts, topk=1, predictor_model=SimpleLSTMClassifier()):
        self.emb = model.shared.to("cuda")
        self.loader_iter = iter(batched_loader)
        self.predictor = predictor_model.to("cuda")
        self.topk = topk

        self.predictor.fc = {k.replace("switch_transformers.", ""): v for k, v in self.predictor.fc.items()}
        self.predictor.y_keys = [k.replace("switch_transformers.", "") for k in self.predictor.y_keys]

        self.expert_module = {}
        for n, m in model.named_modules():
            if n + ".router.classifier" in self.predictor.y_keys:
                self.expert_module[n + ".router.classifier"] = m
        self.expert_status = {k: {"cpu": set(range(n_experts)), "gpu": set()} for k in self.predictor.y_keys}


def main():
    options = argparse.ArgumentParser()
    options.add_argument(
        "--dataset",
        type=str,
        choices=["sst2", "mrpc", "rte", "multirc", "c4_en"],
        help="Run experiments on which dataset",
    )
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
    options.add_argument("--n_experts", type=int, default=None, help="Number of experts in the hash table")
    options.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    options.add_argument("--topk", type=int, default=1, help="Top k experts to use")

    # Parse the arguments
    args = options.parse_args()
    batch_size = args.batch_size
    if (n_experts := args.n_experts) is None:
        # Infer the number of experts from the model name
        n_experts = int(args.model.split("-")[-1])

    print(f"Experiment Configurations: SODA, Dataset: {args.dataset}, Model: {args.model}, Batch Size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model}", cache_dir=f"{BASEDIR}/tmp/")
    model = SwitchTransformersForConditionalGenerationOffload.from_pretrained(
        f"google/{args.model}", cache_dir=f"{BASEDIR}/tmp/"
    )
    model.config.decoder_start_token_id = tokenizer.pad_token_id

    dataloader_config = config.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./default.yaml"))
    dataloader_config.dataset = args.dataset
    dataloader_config.tmp_dir = f"{BASEDIR}/tmp"
    _, test_dataloader = get_mlm_dataloaders(tokenizer, model.config, dataloader_config)
    _, hash_eval_loader = get_mlm_dataloaders(tokenizer, model.config, dataloader_config)
    total_loss = 0
    total_n_batches = 0
    total_accuracy = 0

    predictor = SimpleLSTMClassifier(num_classes=n_experts)
    if os.path.exists(f"{BASEDIR}/data/{args.dataset}/{args.model}/hash_predictor.pt"):
        predictor.load_state_dict(
            torch.load(f"{BASEDIR}/data/{args.dataset}/{args.model}/hash_predictor.pt")["model_state_dict"]
        )
    else:
        logging.warning("Hash predictor not found, will use randomly initialized hash table.")

    deamon = SODAManagerCond(model, hash_eval_loader, n_experts, topk=args.topk, predictor_model=predictor)
    move_to_device(model, "cuda:0")

    for step, batch in enumerate(test_dataloader, start=1):
        if step > 10:
            break
        hash_table, experts_list = deamon.gen_hash()
        deamon.move_experts(experts_list)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                **batch,
                hash_table=[(hash_table, 0, key, args.topk) for key in list(hash_table.keys())[::-1]],
                output_router_logits=False,
            )

        # alculate accuracy:
        predictions = torch.argmax(outputs.logits.view(-1, outputs.logits.size(-1)), dim=1)
        correct_predictions = (predictions == batch["labels"].view(-1)).sum().item()
        total_samples = batch["labels"].view(-1).size(0)
        accuracy = correct_predictions / total_samples

        total_accuracy += accuracy
        total_loss += outputs.loss
        total_n_batches += 1
        print(
            f">>>[Step#{step}] loss: {outputs.loss.item():.6f} ppl: {torch.exp(outputs.loss).item():.2f} accuracy: {accuracy:.2%}"
        )

    ppl = torch.exp(total_loss / total_n_batches).item()
    print(">>>ppl: ", ppl)
    print(f">>>accuracy: {total_accuracy / total_n_batches:.2%}")


if __name__ == "__main__":
    main()
