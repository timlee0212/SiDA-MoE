import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer


def get_metrics(dataset_name):
    if dataset_name == "c4_en":
        metrics = evaluate.load("c4", "en")
    elif dataset_name == "multirc":
        # metrics = evaluate.load("glue", "sst2")
        metrics = load_metric("super_glue", "multirc")
    else:
        metrics = evaluate.load("glue", dataset_name)
    return metrics


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

def tokenize_function(examples, tokenizer, data_column, in_length):
    tokenizer_out = tokenizer(*[examples[col] for col in data_column], return_attention_mask=False)

    input_ids = tokenizer_out["input_ids"]
    concatenated_ids = np.concatenate(input_ids)
    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length
    concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
    result = {"input_ids": concatenated_ids}
    return result


def get_text_cls_loader(
    dataset_name, model_name, tmp_dir, split="validation", hash_loader=False, batch_size=1, hash_step_size=None
):
    assert not hash_loader or hash_step_size is not None, "hash_step_size must be provided if hash_loader is True"
    _task_to_keys = {
        "cola": ("sentence", None, None),
        "mnli": ("premise", "hypothesis", None),
        "mrpc": ("sentence1", "sentence2", None),
        "qnli": ("question", "sentence", None),
        "qqp": ("question1", "question2", None),
        "rte": ("sentence1", "sentence2", None),
        "sst2": ("sentence", None, None),
        "stsb": ("sentence1", "sentence2", None),
        "wnli": ("sentence1", "sentence2", None),
        "multirc": ("paragraph", "question", "answer"),
    }
    
    
    if dataset_name == "multirc":
        # Load MultiRC dataset
        dataset = load_dataset("super_glue", "multirc", cache_dir=f"{tmp_dir}")
    else:
        # Load dataset
        dataset = load_dataset("glue", f"{dataset_name}", cache_dir=f"{tmp_dir}")

    dataset = dataset.shuffle(seed=42)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}", cache_dir=f"{tmp_dir}")
    sentence1_key, sentence2_key, sentence3_key = _task_to_keys[dataset_name]

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if (sentence2_key is None and sentence3_key is None)
            else (examples[sentence1_key], examples[sentence2_key])
            if sentence3_key is None
            else (examples[sentence1_key], examples[sentence2_key], examples[sentence3_key])
        )
        result = tokenizer(
            *args, truncation=True, padding="max_length", max_length=768 if dataset_name == "multirc" else None
        )
        return result

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    pre_proc = lambda inputs: {k: v.to("cuda") for k, v in inputs.items()}

    if dataset_name != "c4_en":
        encoded_dataset = encoded_dataset[split]
    
    if dataset_name == 'multirc':
        encoded_dataset = encoded_dataset.remove_columns('labels')
    
    if dataset_name != 'multirc':
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if dataset_name == 'multirc':
        eval_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size, collate_fn=multirc_data_collator, drop_last=True)
    else:
        eval_loader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size, drop_last=True)

    hash_eval_loader = (
        None if not hash_loader else torch.utils.data.DataLoader(encoded_dataset, batch_size=hash_step_size)
    )

    return eval_loader, hash_eval_loader, pre_proc
