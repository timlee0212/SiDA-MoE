import os, sys, time
from tqdm import tqdm
import torch

from subprocess import Popen, PIPE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))
import config


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    gpus = list(range(torch.cuda.device_count()))
else:
    gpus = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]



# Experiment List
model_list =  ["switch-base-8", "switch-base-64", "switch-base-128", "switch-base-256"]
dataset_list = ["sst2", "mrpc","rte", "multirc"] #"sst2", "mrpc" ,"c4_en" ,"rte", "multirc"

class GPU_Pool:
    def __init__(self, gpus):
        self.gpu_pool = {k: False for k in gpus}
        self.task_queue = []
        self.result_dict = {}

    def add_task(self, task_name, cmdline, environ):
        self.task_queue.append((task_name, cmdline, environ))

    def empty(self):
        return all([v == False for v in self.gpu_pool.values()]) and len(self.task_queue) == 0

    def update(self):
        num_finished = 0
        for k, v in self.gpu_pool.items():
            if isinstance(v, dict):
                if v["proc"].poll() is not None:
                    if v["proc"].returncode != 0:
                        raise RuntimeError(f"{v['proc'].stderr.read().decode()}\nTask {v['name']} failed with return code {v['proc'].returncode}")
                    # This is specialized for this script
                    # We only capture the latency here
                    results = v["proc"].stdout.readlines()
                    for line in results:
                        if ">>>ppl" in line.decode():
                            self.result_dict[v["name"]] = {"ppl": float(line.decode().split(":")[-1])}
                    self.gpu_pool[k] = False
                    num_finished += 1
        for k, v in self.gpu_pool.items():
            if v == False and len(self.task_queue)!=0:
                task_to_run = self.task_queue.pop(0)
                # Assign GPU
                task_to_run[2]["CUDA_VISIBLE_DEVICES"] = str(k)
                self.gpu_pool[k] ={"name":task_to_run[0], "cmdline":task_to_run[1], "environ":task_to_run[2] ,"proc":Popen(task_to_run[1], env=task_to_run[2], shell=True, stdout=PIPE, stderr=PIPE)}
        return num_finished
    
lat_result_dicts = []
total_iterations = len(model_list) * len(dataset_list)
progress_bar = tqdm(total=total_iterations, desc=">>>>>>>>>>>>>Accuracy Tests")

ckpt_dir = os.path.join(os.path.dirname(__file__), "../data")
ckpt_dir = config.ckpt_dir #"/scratch/xiangyj/moe/data"

result_dir_base = os.path.join(os.path.dirname(__file__), "../results")
result_dir_base = config.result_dir_base #"/scratch/xiangyj/moe/results"

acc_script = os.path.join(os.path.dirname(__file__), "../src/accuracy.py")

pool = GPU_Pool(gpus)
for model in model_list:
    pool.add_task(model, f"python {acc_script} --model {model}", environ=os.environ.copy())

while not pool.empty():
    num_done = pool.update()
    progress_bar.update(num_done)
    # Avoid too frequent update
    time.sleep(0.1)

# Save the results
with open(os.path.join(result_dir_base, "accuracy.csv"), "w") as f:
    f.write("model,ppl\n")
    for k, v in pool.result_dict.items():
        f.write(f"{k},{v['ppl']}\n")