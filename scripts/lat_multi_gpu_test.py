import os
import subprocess
from tqdm import tqdm
import torch
import csv, json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/utils"))
import config

if "CUDA_VISIBLE_DEVICES" not in os.environ or len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))<1:
    print("Please set CUDA_VISIBLE_DEVICES to assign this script to more than one GPU.")
    exit(-1)
else:
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"Using GPU {gpus} for the experiment.")

# Experiment List
model_list =  ["switch-base-128"]#,"switch-base-256"
dataset_list = ["multirc","sst2", "mrpc" ,"rte"]#"sst2", "mrpc" finished 64,128,256

# Benchmark Configs
n_warmup_runs = 0
n_benchmark_runs = 5

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

total_iterations = len(model_list) * len(dataset_list) * 2
progress_bar = tqdm(total=total_iterations, desc=">>>>>>>>>>>>>Multi GPU Baseline Test")
tracker = Exp_Tracker(os.path.join(os.path.dirname(__file__), "../results/_lat_mgpu_test_tracker_v100_bs1_bl0_sb128.pt"))

lat_script = os.path.join(os.path.dirname(__file__), "../src/latency.py")
main_script = os.path.join(os.path.dirname(__file__), "../src/main.py")

ckpt_dir = os.path.join(os.path.dirname(__file__), "../data")
ckpt_dir = config.ckpt_dir #"/scratch/xiangyj/moe/data"

result_dir_base = os.path.join(os.path.dirname(__file__), "../results2")
result_dir_base = config.result_dir_base #"/scratch/xiangyj/moe/results"
result_dir_base = "/scratch/xiangyj/moe/results2"


for model in model_list:
    # Run main.py if the checkpoint is not found
    if not os.path.exists(f"{ckpt_dir}/shard_models/{model}"):
        ret = subprocess.run(f"python {main_script} --model {model} --sharding", shell=True)
        if ret.returncode != 0:
            print(f"Error generating sharded weights in {model}")
            exit(-1)
    for dataset in dataset_list:
        for baseline in [0]: # 0,1,2,3,
            local_result_dict = {}
            if baseline == 1:
                cmd = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} python {lat_script} --model {model} --dataset {dataset} --baseline 1 --batch_size 1"
            else:
                cmd = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} deepspeed --num_gpus {len(gpus)} {lat_script} --model {model} --dataset {dataset} --baseline 2 --batch_size 1"
            if not f"lat_{model}_{dataset}_{baseline}" in tracker:
                # We don't need the result of warmup runs
                for _ in range(n_warmup_runs):
                    ret = subprocess.run(cmd, shell=True)
                    if ret.returncode != 0:
                        print(f"Error latency test in {model} {dataset} {baseline}")
                        exit(-1)
                for n in range(n_benchmark_runs):
                    ret = subprocess.run(cmd + f" --running_id {n}", shell=True)
                    if ret.returncode != 0:
                        print(f"Error latency test in {model} {dataset} {baseline}")
                        exit(-1)
            for n in range(n_benchmark_runs):
                with open(os.path.join(result_dir_base, f"{dataset}/{model}/{baseline}_latency_id_{n}.json"), "r") as f:
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
            tracker[f"lat_{model}_{dataset}_{baseline}"] = True
            progress_bar.update(1)

key_list = lat_result_dicts[0].keys()

# Write out the result
with open(os.path.join(result_dir_base, "mgpu_latency_128_v100_bs1_bl0.csv"), "w") as f:
    writer = csv.DictWriter(f, key_list)
    writer.writeheader()
    writer.writerows(lat_result_dicts)

# Delete the tracker file when all experiments are finished
del tracker
os.remove(os.path.join(os.path.dirname(__file__), "../results/_lat_mgpu_test_tracker_v100_bs1_bl0_sb128.pt"))