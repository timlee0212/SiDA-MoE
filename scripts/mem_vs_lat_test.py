import csv
import json
import os
import subprocess
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))
from utils import config

if "CUDA_VISIBLE_DEVICES" not in os.environ or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
    print("Please set CUDA_VISIBLE_DEVICES to assign this script to a single GPU.")
    exit(-1)

# Experiment List
# model_list =  ["switch-base-8", "switch-base-64", "switch-base-128", "switch-base-256"]
model_list =  ["switch-base-128", "switch-base-256"]
# dataset_list = ["sst2", "mrpc", "multirc"]
dataset_list = ["sst2"]
mem_limit_list = [5, 10]
baseline = 1
batch_size_list = [8] + [1,2,4,16]


assert max(mem_limit_list) <= torch.cuda.mem_get_info(0)[0], "Exceed the largest avaliable memory on device"

# Benchmark Configs
n_warmup_runs = 1
n_benchmark_runs = 3


# Save status of the experiment to recover
class Exp_Tracker:
    def __init__(self, tracker_path):
        self.tracker_path = tracker_path
        if os.path.exists(tracker_path):
            self.dict = torch.load(tracker_path)
        else:
            self.dict = {}

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
        torch.save(self.dict, self.tracker_path)

    def __contains__(self, key):
        return key in self.dict


lat_result_dicts = []
key_list = []

total_iterations = len(model_list) * len(dataset_list) * len(mem_limit_list)
progress_bar = tqdm(total=total_iterations, desc=">>>>>>>>>>>>>Latency vs Memory Tests")
tracker = Exp_Tracker(os.path.join(os.path.dirname(__file__), f"../../results/{baseline}_mem_vs_lat_tracker_{model_list[0]}.pt"))

lat_script = os.path.join(os.path.dirname(__file__), "../src/vram_latency.py")
main_script = os.path.join(os.path.dirname(__file__), "../src/main.py")

ckpt_dir = config.ckpt_dir 
result_dir_base = config.result_dir_base

for model in model_list:
    for dataset in dataset_list:
        for batch_size in batch_size_list:
            # Run main.py if the checkpoint is not found
            if not os.path.exists(f"{ckpt_dir}/{dataset}/{model}/activation_validation_large-0.pt"):
                ret = subprocess.run(f"python {main_script} --model {model} --dataset {dataset}", shell=True)
                if ret.returncode != 0:
                    print(f"Error generating checkpoint in {model} {dataset}")
                    exit(-1)
            # Run main.py if the sharded weight is not found
            if not os.path.exists(f"{ckpt_dir}/shard_models/{model}/pytorch_model.bin.index.json"):
                ret = subprocess.run(f"python {main_script} --model {model} --sharding", shell=True)
                if ret.returncode != 0:
                    print(f"Error generating sharded weights in {model}")
                    exit(-1)
            for mem_limit in mem_limit_list:
                local_result_dict = {}
                if not f"lat_{model}_{dataset}_{mem_limit}_{batch_size}" in tracker:
                    # We don't need the result of warmup runs
                    for _ in range(n_warmup_runs):
                        ret = subprocess.run(
                            f"python {lat_script} --model {model} --dataset {dataset} --mem_limit {mem_limit} --baseline {baseline} --batch_size {batch_size}", shell=True
                        )
                        if ret.returncode != 0:
                            print(f"Error latency test in {model} {dataset} {mem_limit}")
                            exit(-1)
                    for n in range(n_benchmark_runs):
                        ret = subprocess.run(
                            f"python {lat_script} --model {model} --dataset {dataset} --mem_limit {mem_limit} --baseline {baseline} --batch_size {batch_size} --running_id {n}",
                            shell=True,
                        )
                        if ret.returncode != 0:
                            print(f"Error latency test in {model} {dataset} {mem_limit}")
                            exit(-1)

                for n in range(n_benchmark_runs):
                    with open(os.path.join(result_dir_base, f"{dataset}/{model}/{baseline}_bs_{batch_size}_latency_id_{n}-tmp.json"), "r") as f:
                        local_result = json.load(f)

                    if len(local_result_dict) == 0:
                        local_result_dict = local_result
                    else:
                        for key in ["communication_time", "loading_time", "forward_time", "expert_selection_time", "end_to_end_time"]:
                            local_result_dict[key] += local_result[key]
                avg_total = local_result_dict["end_to_end_time"]
                for key in ["communication_time", "loading_time", "forward_time", "expert_selection_time"]:
                    local_result_dict[key + "_portion"] = local_result_dict[key + "_portion"] / avg_total
                    local_result_dict[key] /= n_benchmark_runs
                local_result_dict["end_to_end_time"] /= n_benchmark_runs
                local_result_dict.pop("average_loading_time_portion")
                local_result_dict.pop("average_forward_time_portion")

                lat_result_dicts.append(local_result_dict)
                tracker[f"lat_{model}_{dataset}_{mem_limit}"] = True
                progress_bar.update(1)

key_list = lat_result_dicts[0].keys()

# Write out the result
with open(os.path.join(result_dir_base, f"{baseline}_mem_vs_latency.csv"), "w") as f:
    writer = csv.DictWriter(f, key_list)
    writer.writeheader()
    writer.writerows(lat_result_dicts)

# Delete the tracker file when all experiments are finished
del tracker
os.remove(os.path.join(os.path.dirname(__file__), f"../../results/{baseline}_mem_vs_lat_tracker_{model_list[0]}.pt"))
