import os
import subprocess
from tqdm import tqdm
import torch
import csv, json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/utils"))
import config
from helper import Helper as helper

if "CUDA_VISIBLE_DEVICES" not in os.environ or len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))!=1:
    print("Please set CUDA_VISIBLE_DEVICES to assign this script to a single GPU.")
    exit(-1) 

model_list        =  ["switch-base-8"] + ["switch-base-64", "switch-base-128", "switch-base-256"]
dataset_list      =  ["sst2"] #"sst2", "mrpc" ,"rte","c4_en", "multirc" 
batch_size_list   =  [8] + [1, 2, 4, 16]
baseline_list     =  [1] #0,1,2,3
measure_backward  =  1 # whether measure backward, profiling for samsung, in general set as 0

# Benchmark Configs
n_warmup_runs = 0
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

total_iterations = len(model_list) * len(dataset_list) * 3
progress_bar = tqdm(total=total_iterations, desc=">>>>>>>>>>>>>Latency and Memory Tests")
tracker = Exp_Tracker(os.path.join(os.path.dirname(__file__), "../../results/_lat_mem_test_tracker_samsung.pt"))

lat_script = os.path.join(os.path.dirname(__file__), "../src/latency.py")
memory_script = os.path.join(os.path.dirname(__file__), "../src/memory.py")
main_script = os.path.join(os.path.dirname(__file__), "../src/main.py")

ckpt_dir = os.path.join(os.path.dirname(__file__), "../data")
ckpt_dir = config.ckpt_dir #"/scratch/xiangyj/moe/data"

result_dir_base = config.result_dir_base #"/scratch/xiangyj/moe/results"
helper.try_make_dir(result_dir_base)

for model in model_list:
    if not f"mem_{model}" in tracker:
        ret = subprocess.run(f"python {memory_script} --model {model}", shell=True)
        if ret.returncode != 0:
            print(f"Error Memory test in {model}")
            exit(-1)
        tracker[f"mem_{model}"] = True
    progress_bar.update(1)
    for dataset in dataset_list:
        # Run main.py if the checkpoint is not found
        if dataset == "c4_en":
            if not os.path.exists(f"{ckpt_dir}/{dataset}/{model}/activation_validation_large-57.pt"):
                ret = subprocess.run(f"python {main_script} --model {model} --dataset {dataset}", shell=True)
                if ret.returncode != 0:
                    print(f"Error generating checkpoint in {model} {dataset}")
                    exit(-1)
        else:
            if not os.path.exists(f"{ckpt_dir}/{dataset}/{model}/activation_validation_large-0.pt"):
                ret = subprocess.run(f"python {main_script} --model {model} --dataset {dataset}", shell=True)
                if ret.returncode != 0:
                    print(f"Error generating checkpoint in {model} {dataset}")
                    exit(-1)
        for baseline in baseline_list: #0,1,2,3
            for batch_size in batch_size_list:
                local_result_dict = {}
                if not f"lat_{model}_{dataset}_{baseline}_{batch_size}" in tracker:
                    # We don't need the result of warmup runs
                    for _ in range(n_warmup_runs):
                        ret = subprocess.run(f"python {lat_script} --model {model} --dataset {dataset} --baseline {baseline} --batch_size {batch_size} --measure_backward {measure_backward}", shell=True)
                        if ret.returncode != 0:
                            print(f"Error latency test in {model} {dataset} {baseline} {batch_size} {measure_backward}")
                            exit(-1)
                    for n in range(n_benchmark_runs):
                        ret = subprocess.run(f"python {lat_script} --model {model} --dataset {dataset} --baseline {baseline} --batch_size {batch_size} --measure_backward {measure_backward} --running_id {n}", shell=True)
                        if ret.returncode != 0:
                            print(f"Error latency test in {model} {dataset} {baseline} {batch_size} {measure_backward}")
                            exit(-1)
                for n in range(n_benchmark_runs):                         
                    with open(os.path.join(result_dir_base, f"{dataset}/{model}/{baseline}_bs_{batch_size}_latency_id_{n}.json"), "r") as f:
                        local_result = json.load(f)   

                    if len(local_result_dict) == 0:
                        local_result_dict = local_result
                    else:
                        for key in ["gpu_time", "loading_time", "forward_time", "end_to_end_time"]:
                            local_result_dict[key] += local_result[key]
                avg_total = local_result_dict["end_to_end_time"]
                for key in ["gpu_time", "loading_time", "forward_time"]:
                    local_result_dict[key+"_portion"] = local_result_dict[key+"_portion"] / avg_total
                    local_result_dict[key] /= n_benchmark_runs
                local_result_dict["end_to_end_time"] /= n_benchmark_runs
                local_result_dict.pop("average_loading_time_portion")
                local_result_dict.pop("average_forward_time_portion")

                lat_result_dicts.append(local_result_dict)
                tracker[f"lat_{model}_{dataset}_{baseline}_{batch_size}"] = True
                progress_bar.update(1)

key_list = lat_result_dicts[0].keys()

# Write out the result
with open(os.path.join(result_dir_base, f"latency_{model}_A100_single.csv"), "w") as f:
    writer = csv.DictWriter(f, key_list)
    writer.writeheader()
    writer.writerows(lat_result_dicts)

# Delete the tracker file when all experiments are finished
del tracker
os.remove(os.path.join(os.path.dirname(__file__), "../../results/_lat_mem_test_tracker_samsung.pt"))