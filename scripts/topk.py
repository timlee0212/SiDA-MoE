import os
import sys
import pickle
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))

import latency_with_hash
from ft_accuracy import run_end2end
import torch

# The path to save experiment result dict
RESULT_CKPT = os.path.join(os.path.dirname(__file__), "../results/topk_exp.bin")


@dataclass
class ExpArg:
    topk: int = 1
    model: str = "switch-base-8"
    dataset: str = "sst2"
    n_experts: int = None
    # CHANGE ME: confirm before run script
    batch_size: int = 1
    eval_bs: int = 64
    step_size: int = 64
    verbose: bool = False
    running_id: int = 0
    KD: bool = True


result_dict = []
topks = [1, 3, 5]
models = ["switch-base-8"]
dataset = ["mrpc"]

for config in tqdm(product(topks, models, dataset), total=len(topks) * len(models) * len(dataset)):
    args = ExpArg()
    args.topk, args.model, args.dataset = config
    e2e_result = run_end2end(args)
    torch.cuda.empty_cache()
    lat_result = latency_with_hash.main(args)
    result_dict.append(
        {"topk": args.topk, "model": args.model, "dataset": args.dataset, "accuracy": e2e_result, "latency": lat_result}
    )
    torch.cuda.empty_cache()
    pickle.dump(result_dict, open(RESULT_CKPT, "wb"))
