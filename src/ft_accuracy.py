import argparse
import glob
import logging
import os
from collections.abc import Mapping

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_metrics, get_text_cls_loader
from finetune import get_num_labels

from model import SwitchTransformersClassificationModel, SimpleLSTMClassifierSparseAttention, SwitchTransformersClassificationModel_Multirc

from model.predictor import build_sparse_rnn, load_raw_data
from utils import SODAManager, config, move_to_device, monitor_gputil
from utils.hooks import inputHook, routerHook

BASEDIR = config.BASEDIR


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach().cpu().numpy()


def run_end2end(args):
    if args.n_experts is None:
        # Infer the number of experts from the model name
        args.n_experts = int(args.model.split("-")[-1])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    # Check if the finetuned checkpoint exisits
    ft_ckpt = os.path.join(config.ckpt_dir, f"{args.dataset}/{args.model}/finetuned")
    if not os.path.exists(ft_ckpt):
        FileNotFoundError(f"Checkpoint {ft_ckpt} not found. Please run finetuning first.")

    # Load dataset; Warp this as function since we need this for several times
    def prepare_data(soda=False, split="validation"):
        eval_loader, soda_eval_loader, pre_proc = get_text_cls_loader(
            args.dataset,
            args.model,
            f"{BASEDIR}/tmp",
            batch_size=args.eval_bs,
            split=split,
            hash_loader=soda,
            hash_step_size=args.eval_bs,
        )
        metrics = get_metrics(args.dataset)
        return eval_loader, pre_proc, metrics, soda_eval_loader

    eval_loader, pre_proc, metrics, _ = prepare_data()
    # Load finetuned checkpoint
    if args.dataset == "multirc":
        model = SwitchTransformersClassificationModel_Multirc.from_checkpoint(
            get_num_labels(args.dataset),
            MODEL=args.model,
            BASEDIR=BASEDIR,
            ckpt_path=os.path.join(ft_ckpt, "pytorch_model.bin"),
        ).to("cuda")
    else:
        model = SwitchTransformersClassificationModel.from_checkpoint(
            get_num_labels(args.dataset),
            MODEL=args.model,
            BASEDIR=BASEDIR,
            ckpt_path=os.path.join(ft_ckpt, "pytorch_model.bin"),
        ).to("cuda")
    router_hook = routerHook(model, save_dir=ft_ckpt)
    input_hook = inputHook(model, save_dir=ft_ckpt)
    model.eval()

    # Get the metrics of the finetuned model and Generate the checkpoints for the hash table
    for _, batch in tqdm(
        enumerate(eval_loader), total=len(eval_loader), desc="Evaluating finetuned model and get valiadaiton ckpt."
    ):
        batch = pre_proc(batch)
        with torch.no_grad():
            if args.dataset == 'multirc':
                outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch['labels'], idx=batch["idx"])
                outputs = (outputs['logits'], outputs['idx'])
                outputs = nested_detach(outputs)
            else:
                outputs = model(batch["input_ids"], batch["attention_mask"])
            if args.dataset == 'multirc':
                predictions = [{'idx': {'paragraph': q[0], 'question': q[1], 'answer': q[2]},
                                'prediction': np.argmax(p)}
                               for p, q in zip(outputs[0], outputs[1])]
            elif args.dataset != "stsb":
                predictions = torch.argmax(outputs, axis=1)
            else:
                predictions = predictions[:, 0]
            
            if args.dataset == 'multirc':
                metrics.add_batch(predictions=predictions, references=batch["labels"])
            else:
                metrics.add_batch(predictions=predictions, references=batch["label"])
    ft_metrics = metrics.compute()
    torch.save(
        router_hook.activation_dict,
        f"{ft_ckpt}/activation_validation_large-{router_hook.name_ptr}.pt",
    )
    torch.save(input_hook.data_dict, f"{ft_ckpt}/data_validation_large-{input_hook.name_ptr}.pt")

    # Only get train ckpt if it does not exisit
    if (
        len(glob.glob(f"{ft_ckpt}/activation_train_large-*.pt")) == 0
        or len(glob.glob(f"{ft_ckpt}/data_train_large-*.pt")) == 0
    ):
        input_hook.remove()
        router_hook.remove()
        del input_hook, router_hook
        router_hook = routerHook(model, save_dir=ft_ckpt, split="train")
        input_hook = inputHook(model, save_dir=ft_ckpt, split="train")
        train_loader, _, _, _ = prepare_data(soda=False, split="train")
        for _, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Getting train ckpt."):
            batch = pre_proc(batch)
            with torch.no_grad():
                model(batch["input_ids"], batch["attention_mask"])
        torch.save(
            router_hook.activation_dict,
            f"{ft_ckpt}/activation_train_large-{router_hook.name_ptr}.pt",
        )
        torch.save(input_hook.data_dict, f"{ft_ckpt}/data_train_large-{input_hook.name_ptr}.pt")

    # Train the hash Table
    if os.path.exists(f"{ft_ckpt}/hash_predictor.pt"):
        predictor = SimpleLSTMClassifierSparseAttention(num_classes=args.n_experts)
        predictor.load_state_dict(torch.load(f"{ft_ckpt}/hash_predictor.pt")["model_state_dict"])
    else:
        hash_train_loader, hash_eval_loader = load_raw_data(ft_ckpt, soft_target=args.KD)
        predictor, _ = build_sparse_rnn(hash_train_loader, hash_eval_loader, num_experts=args.n_experts, KD=args.KD)
        predictor.save(f"{ft_ckpt}/hash_predictor.pt")
        del hash_eval_loader, hash_train_loader

    # Clean the resources
    input_hook.remove()
    router_hook.remove()
    del (model, eval_loader, metrics, input_hook, router_hook)
    torch.cuda.empty_cache()

    eval_loader, pre_proc, metrics, soda_eval_loader = prepare_data(soda=True)
    model = SwitchTransformersClassificationModel.from_checkpoint(
        get_num_labels(args.dataset),
        MODEL=args.model,
        BASEDIR=BASEDIR,
        ckpt_path=os.path.join(ft_ckpt, "pytorch_model.bin"),
        offloading=True,
    )
    # Evaluate the model with SODA
    manager = SODAManager(model, soda_eval_loader, args.n_experts, topk=args.topk, predictor_model=predictor)

    with monitor_gputil():
        move_to_device(model, "cuda:0")
        model = model.to("cuda:0")
        for _, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating SODA model"):
            hash_table, experts_list = manager.gen_hash(soft_target=args.KD)
            manager.move_experts(experts_list)
            batch = pre_proc(batch)
            with torch.no_grad():
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    hash_table=[(hash_table, 0, key, args.topk) for key in list(hash_table.keys())[::-1]],
                )
                if args.dataset != "stsb":
                    predictions = torch.argmax(outputs, axis=1)
                else:
                    predictions = predictions[:, 0]
                metrics.add_batch(predictions=predictions, references=batch["label"])
    soda_metrics = metrics.compute()
    print("Finetuned model metrics:", ft_metrics)
    print("SODA model metrics:", soda_metrics)

    with open(f"{ft_ckpt}/ft_accuracy.txt", "w") as f:
        f.write(f"Finetuned: \n{ft_metrics}\n")
        f.write(f"SODA: \n{soda_metrics}\n")
    result = {"ft": ft_metrics, "soda": soda_metrics}
    del model, manager, predictor, soda_eval_loader, eval_loader, metrics
    
    # return result


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument(
        "--dataset",
        type=str,
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "multirc"],
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
    options.add_argument("--topk", type=int, default=1, help="Top k experts to use")
    options.add_argument("--eval_bs", type=int, default=64, help="Bacth size used for evaluation")
    options.add_argument("--n_experts", type=int, default=None, help="Number of experts in the hash table")
    options.add_argument("--verbose", action="store_true", help="Show debug outputs")
    options.add_argument("--KD", action="store_true", help="Whether to use knowledge distillation")
    args = options.parse_args()
    run_end2end(args)
