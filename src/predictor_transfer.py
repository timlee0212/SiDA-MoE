import argparse
import glob
import logging
import os

import torch
from tqdm import tqdm

from dataset import get_metrics, get_text_cls_loader
from finetune import get_num_labels
from model import SimpleLSTMClassifierSparseAttention, SwitchTransformersClassificationModel
from model.predictor import build_sparse_rnn, load_raw_data
from utils import SODAManager, config, move_to_device
from utils.hooks import inputHook, routerHook

BASEDIR = config.BASEDIR

options = argparse.ArgumentParser()
options.add_argument(
    "--src",
    type=str,
    #choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
    choices=["mrpc", "sst2"],
    help="Run experiments on which dataset",
)
options.add_argument(
    "--tgt",
    type=str,
    #choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
    choices=["mrpc", "sst2"],
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
options.add_argument("--topk", type=int, default=1, help="Top k experts for evaluation")
options.add_argument("--KD", action="store_true", help="Whether to use knowledge distillation")
args = options.parse_args()
if args.n_experts is None:
    # Infer the number of experts from the model name
    args.n_experts = int(args.model.split("-")[-1])


# Load dataset; Warp this as function since we need this for several times
def prepare_data(dataset, split="validation"):
    eval_loader, _, pre_proc = get_text_cls_loader(
        dataset,
        args.model,
        f"{BASEDIR}/tmp",
        batch_size=64,
        split=split,
        hash_loader=False,
        hash_step_size=64,
    )
    return eval_loader, pre_proc


eval_loader, pre_proc = prepare_data(args.src)
ft_ckpt = os.path.join(config.ckpt_dir, f"{args.src}/{args.model}/finetuned")
# Load finetuned checkpoint
model = SwitchTransformersClassificationModel.from_checkpoint(
    get_num_labels(args.src),
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
        outputs = model(batch["input_ids"], batch["attention_mask"])
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
    train_loader, _ = prepare_data(args.src, split="train")
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
# hash_train_loader, hash_eval_loader = load_raw_data(ft_ckpt, soft_target=args.KD)
# predictor, _ = build_sparse_rnn(hash_train_loader, hash_eval_loader, num_experts=args.n_experts, KD=args.KD)
# del hash_eval_loader, hash_train_loader
predictor = SimpleLSTMClassifierSparseAttention(num_classes=args.n_experts)
predictor.load_state_dict(torch.load(f"{ft_ckpt}/hash_predictor.pt")["model_state_dict"])
predictor = predictor.to("cuda")
# Clean the resources
input_hook.remove()
router_hook.remove()
del (model, eval_loader, input_hook, router_hook)
torch.cuda.empty_cache()

# Evaluate the transfer performance
trans_ckpt = os.path.join(config.ckpt_dir, f"{args.tgt}/{args.model}/finetuned")
if (
    len(glob.glob(f"{trans_ckpt}/activation_train_large-*.pt")) == 0
    or len(glob.glob(f"{trans_ckpt}/data_train_large-*.pt")) == 0
):
    # Load finetuned checkpoint
    model = SwitchTransformersClassificationModel.from_checkpoint(
        get_num_labels(args.tgt),
        MODEL=args.model,
        BASEDIR=BASEDIR,
        ckpt_path=os.path.join(trans_ckpt, "pytorch_model.bin"),
    ).to("cuda")
    router_hook = routerHook(model, save_dir=trans_ckpt)
    input_hook = inputHook(model, save_dir=trans_ckpt)
    model.eval()
    # Generate the target checkpoint if it does not exists
    eval_loader, pre_proc = prepare_data(args.tgt)
    for _, batch in tqdm(
        enumerate(eval_loader), total=len(eval_loader), desc="Evaluating finetuned model and get valiadaiton ckpt."
    ):
        batch = pre_proc(batch)
        with torch.no_grad():
            outputs = model(batch["input_ids"], batch["attention_mask"])
    torch.save(
        router_hook.activation_dict,
        f"{trans_ckpt}/activation_validation_large-{router_hook.name_ptr}.pt",
    )
    torch.save(input_hook.data_dict, f"{trans_ckpt}/data_validation_large-{input_hook.name_ptr}.pt")
_, hash_eval_loader = load_raw_data(trans_ckpt, soft_target=args.KD)
trans_acc = predictor.evaluate(hash_eval_loader, topK=args.topk, KD=args.KD)
print(trans_acc)
