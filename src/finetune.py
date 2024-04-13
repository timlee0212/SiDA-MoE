import argparse
import os

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, Trainer, TrainingArguments

from model import SwitchTransformersClassificationModel, SwitchTransformersClassificationModel_Multirc
from utils import config

os.environ["WANDB_DISABLED"] = "true"
BASEDIR = config.BASEDIR


def parse_args(verbose=True):
    parser = argparse.ArgumentParser("Finetune MOE for GLUE classification")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--benchmark", type=str, required=True, choices=["glue", "super_glue"])
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["multirc", "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
    )
    parser.add_argument("--model", type=str, required=True, default="switch-base-8")
    parser.add_argument("--batch_size", type=int, required=True, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    args = parser.parse_args()

    if verbose:
        for k, v in args.__dict__.items():
            print(f"{k}={v}")

    return args


def get_num_labels(task):
    task_to_num_labels = {
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
        "qnli": 2,
        "qqp": 2,
        "rte": 2,
        "sst2": 2,
        "stsb": 1,
        "wnli": 2,
        "multirc": 2,
    }
    return task_to_num_labels[task]


def get_tokenize_function(task, model):
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model}")

    if task == "cola" or task == "sst2":

        def tokenize_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    elif task == "mnli":

        def tokenize_function(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)

    elif task == "mrpc" or task == "stsb" or task == "wnli" or task == "rte":

        def tokenize_function(examples):
            return tokenizer(
                examples["sentence1"], examples["sentence2"], padding="max_length", max_length=128, truncation=True
            )

    elif task == "qnli":

        def tokenize_function(examples):
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)

    elif task == "qqp":

        def tokenize_function(examples):
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)

    elif task == "multirc":

        def tokenize_function(examples):
            return tokenizer(
                examples["paragraph"],
                examples["question"],
                examples["answer"],
                padding="max_length",
                max_length=768,
                truncation=True,
            )

    else:
        raise ValueError("Task is not supported!")

    return tokenize_function


def get_train_eval_splits(tokenized_datasets, task):
    train_key = "train"
    if task == "mnli":
        eval_key = "validation_matched"
    else:
        eval_key = "validation"
    return tokenized_datasets[train_key], tokenized_datasets[eval_key]

def get_compute_metrics(metric, task):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task == 'multirc':
            predictions = [{'idx': {'paragraph': q[0], 'question': q[1], 'answer': q[2]},
                            'prediction': np.argmax(p)}
                           for p, q in zip(predictions[0], predictions[1])]
        elif task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def multirc_data_collator(features):
    import torch
    from collections.abc import Mapping

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k == "idx":
            batch[k] = torch.tensor(list(map(lambda f: (f[k]['paragraph'], f[k]['question'], f[k]['answer']), features)))
        elif k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


if __name__ == "__main__":
    args = parse_args()
    print(args)
    working_dir = os.path.join(config.BASEDIR, "data", args.task, args.model, "finetuned")
    print(f"Save model to {working_dir}")
    os.makedirs(working_dir, exist_ok=True)

    # Load dataset and metric
    dataset = load_dataset(args.benchmark, args.task, cache_dir=f"{config.BASEDIR}/tmp/")
    metric = load_metric(args.benchmark, args.task)

    # Load model
    if args.task != "multirc":
        model = SwitchTransformersClassificationModel(
            MODEL=args.model, BASEDIR=config.BASEDIR, num_labels=get_num_labels(args.task)
        )
    else:
        model = SwitchTransformersClassificationModel_Multirc(
            MODEL=args.model, BASEDIR=config.BASEDIR, num_labels=get_num_labels(args.task)
        )

    model.train()

    # # Freeze all layers except the classification head
    # for param in model.switch_transformers.parameters():
    #     param.requires_grad = False

    tokenized_datasets = dataset.map(get_tokenize_function(args.task, args.model), batched=True)
    # Remove column for multirc
    if args.task == "multirc":
        tokenized_datasets = tokenized_datasets.remove_columns("labels")

    train_dataset, eval_dataset = get_train_eval_splits(tokenized_datasets, args.task)

    train_args = TrainingArguments(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=working_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=20000,
        logging_steps=200,
        save_steps=200,
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        save_strategy = "steps",
        save_total_limit=2,
        local_rank=args.local_rank,
        half_precision_backend="cuda_amp",
        ddp_backend="nccl",
        # HF automatically use standard optimizer for embedding layer
        optim="adamw_bnb_8bit",
        ddp_find_unused_parameters=True,
        auto_find_batch_size=True,
        # Reduce the memory usage but slower
        # gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_a" if args.task == "multirc" else None,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(metric, args.task),
        data_collator=multirc_data_collator if args.task == "multirc" else None,
    )

    trainer.train()
    trainer.save_model()
    eval_result = trainer.evaluate()

    print(">>>RESULT:", eval_result)
