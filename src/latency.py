import argparse
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/utils"))

import deepspeed
import torch
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_in_model
from transformers import modeling_utils

from dataset import get_text_cls_loader
from model import SwitchTransformersClassificationModel
from utils import config, get_module_memory_usage, load_oracle_hash, move_to_device, monitor_gputil, add_timing_hooks, TimedSwitchTransformersTop1Router

BASEDIR = config.BASEDIR
RESULT_PATH_BASE = config.result_dir_base #os.path.join(BASEDIR, "results", f"./{DATASET}/{MODEL}/")
logger = logging.getLogger("lat_test")
logger.addHandler(logging.StreamHandler(sys.stdout))

# Disable the warnings related to weight
modeling_utils.logger.setLevel(logging.ERROR)
SPLIT = "validation"  # test validation


def time_forward(start, end, eval_loader, model, preproccss_func, gpu_time, topk=1, hash_table=None, DATASET=None, measure_backward=False):
    # Sanity check
    print(measure_backward)
    if measure_backward and hash_table is not None:
        assert False, "measure_backward is not compatable with baseline 0"
    
    add_timing_hooks(model)
    start.record()

    total_loading_time  = 0
    total_forward_time  = 0
    total_backward_time = 0
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
        
        labels = inputs["label"] if measure_backward else None
        batch_size = inputs["input_ids"].size(0)  # assuming your inputs dict contains "input_ids" key

        start_step.record()
        if hash_table is not None:
            with torch.no_grad():
                output = model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    hash_table=[(hash_table, step, key, topk) for key in list(hash_table.keys())[::-1]],
                )
        else:
            with torch.set_grad_enabled(bool(measure_backward)):
                output = model(inputs["input_ids"], inputs["attention_mask"], labels=labels)
        end_step.record()
        torch.cuda.synchronize()
        forward_time = start_step.elapsed_time(end_step) / 1000.0
        total_samples += batch_size
        total_forward_time += forward_time
        
        if measure_backward:
            start_step.record()
            output[0].backward()
            end_step.record()
            torch.cuda.synchronize()
            backward_time = start_step.elapsed_time(end_step) / 1000.0
            total_backward_time += backward_time     
        
        del inputs
        # Throughput for each forward pass
        throughput_each_forward = batch_size / forward_time
        logger.debug(f"Throughput for step {step}: {throughput_each_forward} samples/second")

        if DATASET == "c4_en" and step > len(eval_loader) / 10:
            break

    end.record()
    torch.cuda.synchronize()
    end_to_end_time = start.elapsed_time(end) / 1000.0

    # Throughput for the entire process
    total_throughput = total_samples / end_to_end_time
    logger.debug(f"Total throughput: {total_throughput} samples/second")

    gpu_time_portion = (gpu_time / end_to_end_time) * 100
    loading_time_portion = (total_loading_time / end_to_end_time) * 100
    forward_time_portion = (total_forward_time / end_to_end_time) * 100
    backward_time_portion = (total_backward_time / end_to_end_time) * 100
    switch_transformers_time = sum([module.total_time for module in model.modules() if isinstance(module, TimedSwitchTransformersTop1Router)])
    
    logger.debug(f"Time for moving model to GPU: {gpu_time} seconds, portion: {gpu_time_portion}%")
    logger.debug(f"Total time for loading data: {total_loading_time} seconds, portion: {loading_time_portion}%")
    logger.debug(f"Total time for forward pass: {total_forward_time} seconds, portion: {forward_time_portion}%")
    logger.debug(f"Total time for backward pass: {total_backward_time} seconds")
    logger.debug(f"Total time for SwitchTransformersTop1Router modules: {switch_transformers_time} ms")
    logger.debug(f"End to end time: {end_to_end_time} seconds")

    average_loading_time_portion = (total_loading_time / (total_loading_time + total_forward_time + total_backward_time)) * 100
    average_forward_time_portion = (total_forward_time / (total_loading_time + total_forward_time + total_backward_time)) * 100
    average_backward_time_portion = (total_backward_time / (total_loading_time + total_forward_time + total_backward_time)) * 100

    logger.debug(f"Average time portion for loading data in each forward pass: {average_loading_time_portion}%")
    logger.debug(f"Average time portion for forward pass in each forward pass: {average_forward_time_portion}%")
    logger.debug(f"Average time portion for backward pass in each forward pass: {average_backward_time_portion}%") 
    
    output_dict = {
        "total_throughput": total_throughput,
        "gpu_time": gpu_time,
        "gpu_time_portion": gpu_time_portion,
        "loading_time": total_loading_time,
        "loading_time_portion": loading_time_portion, 
        "forward_time": total_forward_time,
        "forward_time_portion": forward_time_portion,
        "backward_time": total_backward_time,
        "backward_time_portion": backward_time_portion,
        "expert_selection_time": switch_transformers_time,
        "end_to_end_time": end_to_end_time, 
        "average_loading_time_portion": average_loading_time_portion,
        "average_forward_time_portion": average_forward_time_portion,
        "num_batches": step+1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }

    return output_dict


def main():
    options = argparse.ArgumentParser()
    options.add_argument("--baseline", type=int, choices=[0, 1, 2, 3], help="Run the baseline")
    options.add_argument(
        "--dataset",
        type=str,
        choices=["sst2", "mrpc", "rte", "multirc", "c4_en"],
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
    options.add_argument("--trace", action="store_true", help="Trace the model")
    options.add_argument("--measure_backward", type=int, default=0, help="Whether to measure the backward time for the model")
    options.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    options.add_argument("--verbose", action="store_true", help="Show debug outputs")
    options.add_argument("--running_id", type=int, default=0, help="refers the id of experiments running")
    options.add_argument("--local_rank", type=int, default=0, help="local rank for mp runs")

    args = options.parse_args()
    BASELINE = args.baseline
    DATASET = args.dataset
    MODEL = args.model
    REPORT_PATH_BASE = os.path.join(RESULT_PATH_BASE, f"./{DATASET}/{MODEL}/")
    
    batch_size = args.batch_size

    gpus = [args.local_rank]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    multi_gpu = len(gpus) > 1

    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    logger.info(
        f"Experiment Configurations: Baseline: {BASELINE}, Dataset: {DATASET}, Model: {MODEL}, Batch Size: {batch_size}"
    )
    logger.info(f"Result will be written to {REPORT_PATH_BASE}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Save profiling reports
    if not os.path.exists(REPORT_PATH_BASE):
        os.makedirs(os.path.dirname(REPORT_PATH_BASE), exist_ok=True)

    DATA_PATH = f"{BASEDIR}/data/{DATASET}/{MODEL}/"
    logger.debug(DATA_PATH)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=False)

    # Load dataset
    eval_loader, _, pre_proc = get_text_cls_loader(
        DATASET, MODEL, f"{BASEDIR}/tmp", split=SPLIT, hash_loader=False, batch_size=batch_size
    )
    
    if BASELINE == 1:
        sharded_path = os.path.join(BASEDIR, f"data/shard_models/{MODEL}")
        if multi_gpu and not os.path.exists(sharded_path):
            logging.error("Sharded Weight does not exisit. Run main.py -sharding first!")

        model = SwitchTransformersClassificationModel(
            num_labels=2, MODEL=MODEL, BASEDIR=BASEDIR, meta=multi_gpu, offloading=False
        )

        if multi_gpu:
            memory_limit = {k: "28GiB" for k in gpus}
            memory_limit["cpu"] = "1024GiB"
            device_map = infer_auto_device_map(
                model, max_memory=memory_limit, no_split_module_classes=["SwitchTransformersLayerFF"]
            )
            print(device_map)
            device_map_cpu = {k: "cpu" for k in device_map.keys()}
            load_checkpoint_in_model(model, checkpoint=sharded_path, device_map=device_map_cpu)

        if not multi_gpu:
            model.cpu()
        with monitor_gputil():
            start.record()
            if not multi_gpu:
                move_to_device(model, "cuda")
            else:
                model = dispatch_model(model, device_map=device_map)
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / 1000.0

            output_dict = time_forward(
                start=start, end=end, eval_loader=eval_loader, model=model, gpu_time=gpu_time, preproccss_func=pre_proc, measure_backward=args.measure_backward
            )

    if BASELINE in [2, 3]:
        import time

        config = {
            "train_batch_size": 1,  # Dummy value
            "train_micro_batch_size_per_gpu": 1,  # Dummy value
            "steps_per_print": 9999999999,  # Dummy value
            "wall_clock_breakdown": False,
            "fp16": {
                "enabled": False,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "number_checkpoints": 2,
                "contiguous_memory_optimization": True,
                "synchronize_checkpoint_boundary": False,
            },
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
        }
        model = SwitchTransformersClassificationModel(num_labels=2, MODEL=MODEL, tutel=BASELINE == 3, BASEDIR=BASEDIR)
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        model.cpu()

        # CUDA event does not work under this distributed setting; We have to use time.time() instead
        with monitor_gputil():
            start_t = time.time()
            if multi_gpu:
                ds_engine = deepspeed.init_inference(
                    model,
                    mp_size=world_size,
                    dtype=torch.float,
                    checkpoint=None,
                    replace_with_kernel_inject=True,
                )
                model = ds_engine.module
            else:
                model, _, _, _ = deepspeed.initialize(config_params=config, model=model)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_t  # In seconds
            output_dict = time_forward(
                start=start, end=end, eval_loader=eval_loader, model=model, gpu_time=gpu_time, preproccss_func=pre_proc, measure_backward=args.measure_backward
            )

    if BASELINE == 0:
        hash_table = load_oracle_hash(
            f"{BASEDIR}/data/{DATASET}/{MODEL}", ratio=0.1 if DATASET == "c4_en" else None, split=SPLIT, topk=args.topk
        )

        model = SwitchTransformersClassificationModel(2, MODEL=MODEL, BASEDIR=BASEDIR, offloading=True)
        offload_dict = {}
        for n, m in model.named_modules():
            if n + ".router.classifier" in hash_table.keys():
                offload_dict[n] = m

        model.cpu()
        with monitor_gputil():
            start.record()
            move_to_device(model, "cuda:0")

            for key, val in offload_dict.items():
                offload_experts = torch.unique(hash_table[key + ".router.classifier"])
                for expert_id in offload_experts:
                    val.experts[f"expert_{int(expert_id)}"].to("cuda:0")
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / 1000

            output_dict = time_forward(
                start=start,
                end=end,
                eval_loader=eval_loader,
                model=model,
                gpu_time=gpu_time,
                preproccss_func=pre_proc,
                hash_table=hash_table,
                DATASET=DATASET,
            )

    # Save the result is any of the baseline runs
    logger.debug(output_dict)

    if "output_dict" in locals():
        # Attach the experiment configurations
        output_dict["baseline"] = BASELINE
        output_dict["dataset"] = DATASET
        output_dict["model"] = MODEL
        output_dict["model_mem"] = get_module_memory_usage(model) / 1024**3  # In GB

        report_name = f"{BASELINE}_bs_{batch_size}_latency_id_{args.running_id}.json"
        print(f"Report path: {os.path.join(REPORT_PATH_BASE, report_name)}")
        if os.path.exists(report_name):
            logger.warning(f"Report {report_name} already exists, will be overwritten!")
        with open(os.path.join(REPORT_PATH_BASE, report_name), "w") as outfile:
            json.dump(output_dict, outfile)
            outfile.write("\n")

if __name__ == "__main__":
    main()

    # switch_transformers
    #    |--- shared
    #    |--- encoder
    #            |--- [embed_tokens]: 0.122559 GB
    #            |--- [block]: 48.756044 GB
    #                    | --- [0]: 0.046885 GB
    #                    | --- [1]: 4.016121 GB
    #                    | --- [2]: 0.046883 GB
    #                    | --- [3]: 4.016121 GB
    #                    | --- [4]: 0.046883 GB
    #                    | --- [5]: 4.016121 GB
    #                    | --- [6]: 0.046883 GB
    #                    | --- [7]: 4.016121 GB
    #                    | --- [8]: 0.046883 GB
    #                    | --- [9]: 4.016121 GB
    #                    | --- [10]: 0.046883 GB
    #                    | --- [11]: 4.016121 GB
    #                    | --- [12]: 0.046883 GB
    #                    | --- [13]: 4.016121 GB
    #                    | --- [14]: 0.046883 GB
    #                    | --- [15]: 4.016121 GB
    #                    | --- [16]: 0.046883 GB
    #                    | --- [17]: 4.016121 GB
    #                    | --- [18]: 0.046883 GB
    #                    | --- [19]: 4.016121 GB
    #                    | --- [20]: 0.046883 GB
    #                    | --- [21]: 4.016121 GB
    #                    | --- [22]: 0.046883 GB
    #                    | --- [23]: 4.016121 GB
    #            |--- [final_layer_norm]: 0.000004 GB
    #            |--- [dropout]: 0.000000 GB

    # for name, module in model.switch_transformers.encoder.block.named_children():
    #     print(f'[{name}]: {get_module_memory_usage(module)/ 1024**3:.6f} GB')

    # SwitchTransformersSparseMLP(nn.Module) -> input: hidden_states; output: hidden_states, (router_logits, expert_index)
    # SwitchTransformersLayerFF(nn.Module)   -> in-built two kinds of mlps; input: hidden_states; output: output hidden state
    # Only need to change the code in SwitchTransformersBlock, such that it will replace the 'SwitchTransformersLayerFF' by \
    # specific expert given the hasing table
