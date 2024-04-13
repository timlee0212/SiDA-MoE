import argparse
import datetime
import json
import logging
import math
import os
import sys

import torch
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_in_model
from transformers import modeling_utils

from dataset import get_text_cls_loader
from model import SwitchTransformersClassificationModel
from utils import config, get_module_memory_usage, load_oracle_hash, add_timing_hooks, TimedSwitchTransformersTop1Router

# os.environ["NCCL_P2P_DISABLE"]="1"
BASEDIR = config.BASEDIR

logger = logging.getLogger("lat_test")
logger.addHandler(logging.StreamHandler(sys.stdout))
# Disable the warnings related to weight
modeling_utils.logger.setLevel(logging.ERROR)
SPLIT = "validation"  # test validation


def get_module_from_name(base_model: torch.nn.Module, module_path: str):
    mod = base_model
    for name in module_path.split("."):
        if name.isdigit():
            mod = mod[int(name)]
        else:
            mod = getattr(mod, name)
    return mod


def get_soda_device_map(device_map, model, mem_limit, hash_table):
    # Record modules other than experts
    aux_dict = {}
    for k, _ in device_map.items():
        # Reset all experts on cpu
        if not "experts" in k:
            device_map[k] = 0
            if k not in aux_dict:
                # Recurisvely reach the target module
                # Need exec here since aux_module is a string
                aux_dict[k] = get_module_memory_usage(get_module_from_name(model, k))
        else:
            device_map[k] = "cpu"

    aux_size = sum(aux_dict.values()) / 1024**3
    experts_mem = mem_limit - aux_size
    assert (
        experts_mem >= 0
    ), f"No Memory left for experts! {mem_limit} GiB assigned, {aux_size} GiB used by other modules."

    total_experts_size = 0
    experts_on_cpu = 0
    experts_on_gpu = 0
    for layer in hash_table.keys():
        offload_experts = torch.unique(hash_table[layer])
        for expert_id in offload_experts:
            expert_name = layer.replace("router.classifier", "experts") + f".expert_{expert_id}"
            expert_size = get_module_memory_usage(get_module_from_name(model, expert_name)) / 1024**3
            total_experts_size += expert_size
            if total_experts_size <= experts_mem:
                logger.debug(f"----- Assigning {layer+'.expert_' + str(expert_id)}:{expert_size}GiB to GPU")
                device_map[expert_name + ".wi"] = 0
                device_map[expert_name + ".wo"] = 0
                # For efficiency, we should skip the rest of the loop
                # But we continue anyway for debugging purpose
                experts_on_gpu += 1
            else:
                experts_on_cpu += 1
    logger.debug(f"--- Assigning {experts_on_gpu} experts to GPU, {experts_on_cpu} experts to CPU")
    return device_map


def time_forward(start, end, eval_loader, model, preproccss_func, gpu_time, hash_table=None):
    add_timing_hooks(model)
    start.record()

    total_loading_time = 0
    total_forward_time = 0
    total_samples = 0  # total number of samples

    for step, inputs in enumerate(eval_loader):
        start_step = torch.cuda.Event(enable_timing=True)
        end_step = torch.cuda.Event(enable_timing=True)
        start_step.record()
        inputs = preproccss_func(inputs)
        end_step.record()
        torch.cuda.synchronize()
        loading_time = start_step.elapsed_time(end_step) / 1000.0
        total_loading_time += loading_time

        batch_size = inputs["input_ids"].size(0)  # assuming your inputs dict contains "input_ids" key

        start_step.record()
        if hash_table is not None:
            with torch.no_grad():
                output = model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    hash_table=[(hash_table, step, key, 1) for key in list(hash_table.keys())[::-1]],
                )
        else:
            with torch.no_grad():
                output = model(inputs["input_ids"], inputs["attention_mask"])
        end_step.record()
        torch.cuda.synchronize()
        forward_time = start_step.elapsed_time(end_step) / 1000.0
        total_samples += batch_size
        total_forward_time += forward_time

        # Throughput for each forward pass
        throughput_each_forward = batch_size / forward_time
        logger.debug(f"Throughput for step {step}: {throughput_each_forward} samples/second")

    end.record()
    torch.cuda.synchronize()
    end_to_end_time = start.elapsed_time(end) / 1000.0

    # Throughput for the entire process
    total_throughput = total_samples / end_to_end_time
    logger.debug(f"Total throughput: {total_throughput} samples/second")
    
    expert_selection_time = sum([module.total_time for module in model.modules() if isinstance(module, TimedSwitchTransformersTop1Router)])


    gpu_time_portion = (gpu_time / end_to_end_time) * 100
    loading_time_portion = (total_loading_time / end_to_end_time) * 100
    forward_time_portion = (total_forward_time - expert_selection_time / end_to_end_time) * 100
    expert_selection_time_portion = (total_forward_time - expert_selection_time / end_to_end_time) * 100

    logger.debug(f"Time for moving model to GPU: {gpu_time} seconds, portion: {gpu_time_portion}%")
    logger.debug(f"Total time for loading data: {total_loading_time} seconds, portion: {loading_time_portion}%")
    logger.debug(f"Total time for forward pass: {total_forward_time} seconds, portion: {forward_time_portion}%")
    logger.debug(f"End to end time: {end_to_end_time} seconds")

    average_loading_time_portion = (total_loading_time / (total_loading_time + total_forward_time)) * 100
    average_forward_time_portion = (total_forward_time / (total_loading_time + total_forward_time)) * 100

    logger.debug(f"Average time portion for loading data in each forward pass: {average_loading_time_portion}%")
    logger.debug(f"Average time portion for forward pass in each forward pass: {average_forward_time_portion}%")

    output_dict = {
        "total_throughput": total_throughput,
        "communication_time": gpu_time,
        "communication_time_portion": gpu_time_portion,
        "loading_time": total_loading_time,
        "loading_time_portion": loading_time_portion,
        "forward_time": total_forward_time - expert_selection_time,
        "forward_time_portion": forward_time_portion,
        "expert_selection_time": expert_selection_time,
        "expert_selection_time_portion": expert_selection_time_portion,
        "end_to_end_time": end_to_end_time,
        "average_loading_time_portion": average_loading_time_portion,
        "average_forward_time_portion": average_forward_time_portion,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }

    return output_dict


def main():
    options = argparse.ArgumentParser()
    options.add_argument(
        "--dataset", type=str, choices=["sst2", "mrpc", "rte", "multirc"], help="Run experiments on which dataset"
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
    options.add_argument("--baseline", type=int, choices=[0, 1], help="Run baseline or not")
    options.add_argument("--mem_limit", type=float, default=0, help="Upper limit of GPU memory.")
    options.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    options.add_argument("--verbose", action="store_true", help="Show debug outputs")
    options.add_argument("--running_id", type=int, default=0, help="refers the id of experiments running")

    args = options.parse_args()
    DATASET = args.dataset
    MODEL = args.model
    REPORT_PATH_BASE = os.path.join(config.result_dir_base, f"./{DATASET}/{MODEL}/")
    batch_size = args.batch_size

    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    logger.info(
        f"Experiment Configurations: Baseline {args.baseline}, VARM Limit {args.mem_limit}GiB, Dataset: {DATASET}, Model: {MODEL}, Batch Size: {batch_size}"
    )
    logger.info(f"Result will be written to {REPORT_PATH_BASE}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Save profiling reports
    if not os.path.exists(REPORT_PATH_BASE):
        os.makedirs(REPORT_PATH_BASE, exist_ok=True)

    DATA_PATH = f"{BASEDIR}/data/{DATASET}/{MODEL}/"
    logger.debug(DATA_PATH)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=False)

    eval_loader, _, prec_func = get_text_cls_loader(DATASET, MODEL, f"{BASEDIR}/tmp/", batch_size=batch_size)

    sharded_path = os.path.join(BASEDIR, f"data/shard_models/{MODEL}")
    if not os.path.exists(sharded_path):
        logger.error("Sharded Weight does not exisit. Run main.py -sharding first!")

    hash_table = load_oracle_hash(f"{BASEDIR}/data/{DATASET}/{MODEL}/")

    model = SwitchTransformersClassificationModel(
        2, MODEL=MODEL, BASEDIR=BASEDIR, meta=True, offloading=args.baseline == 0
    )

    # By default, hf will combine and simplify device map
    # So we have to load it from the sharded checkpoints
    weight_map = json.load(open(os.path.join(sharded_path, "pytorch_model.bin.index.json")))["weight_map"]
    # Get the module name instead of weight map
    device_map = {}
    for k in weight_map.keys():
        module_name = ".".join(k.split(".")[:-1])
        device_map[module_name] = "cpu"

    device_map_cpu = {"": "cpu"}
    load_checkpoint_in_model(model, checkpoint=sharded_path, device_map=device_map_cpu)

    device_map = (
        infer_auto_device_map(model, max_memory={0: f"{math.floor(args.mem_limit)}GiB", "cpu": "1024GiB"})
        if args.baseline
        else get_soda_device_map(device_map, model, args.mem_limit, hash_table)
    )
    print(device_map)
    start.record()
    model = dispatch_model(model, device_map=device_map)
    end.record()
    torch.cuda.synchronize()

    gpu_time = start.elapsed_time(end) / 1000

    output_dict = time_forward(
        start=start,
        end=end,
        eval_loader=eval_loader,
        model=model,
        gpu_time=gpu_time,
        preproccss_func=prec_func,
        hash_table=hash_table,
    )

    # Save the result is any of the baseline runs
    logger.debug(output_dict)
    if "output_dict" in locals():
        # Attach the experiment configurations
        output_dict["mem_limit"] = args.mem_limit
        output_dict["dataset"] = DATASET
        output_dict["model"] = MODEL

        output_dict["model_mem"] = get_module_memory_usage(model) / 1024**3  # In GB

        report_name = f"{args.baseline}_latency_id_{args.running_id}.json"
        if os.path.exists(report_name):
            logger.warning(f"Report {report_name} already exists, will be overwritten!")

        # For communicate with script
        print(os.path.join(REPORT_PATH_BASE, f"{args.baseline}_bs_{args.batch_size}_latency_id_{args.running_id}-tmp.json"))
        with open(os.path.join(REPORT_PATH_BASE, f"{args.baseline}_bs_{args.batch_size}_latency_id_{args.running_id}-tmp.json"), "w") as outfile:
            json.dump(output_dict, outfile)
            outfile.write("\n")

        # For persistence
        with open(
            os.path.join(REPORT_PATH_BASE, f"{args.baseline}_bs_{args.batch_size}_latency_id_{args.running_id}_{args.mem_limit}.json"), "w"
        ) as outfile:
            json.dump(output_dict, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    main()

    # SwitchTransformersSparseMLP(nn.Module) -> input: hidden_states; output: hidden_states, (router_logits, expert_index)
    # SwitchTransformersLayerFF(nn.Module)   -> in-built two kinds of mlps; input: hidden_states; output: output hidden state
    # Only need to change the code in SwitchTransformersBlock, such that it will replace the 'SwitchTransformersLayerFF' by \
    # specific expert given the hasing table
