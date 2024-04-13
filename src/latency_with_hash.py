import argparse
import datetime
import json
import logging
import os
import queue
import sys

import torch
from transformers import modeling_utils

from dataset import get_text_cls_loader

from model import SimpleLSTMClassifierSparseAttention, SwitchTransformersClassificationModel
from utils import SODAManager, SODAThread, config, move_to_device, monitor_gputil


BASEDIR = config.BASEDIR
logger = logging.getLogger("lat_test")
logger.addHandler(logging.StreamHandler(sys.stdout))
# Disable the warnings related to weight
modeling_utils.logger.setLevel(logging.ERROR)
SPLIT = "validation"  # test validation

def main(args):
    assert (
        args.step_size % args.batch_size == 0
    ), f"step_size {args.step_size} is not divisible by batch_size {args.batch_size} "
    DATASET = args.dataset
    MODEL = args.model
    REPORT_PATH_BASE = os.path.join(BASEDIR, "hash_results", f"./{DATASET}/{MODEL}/")
    batch_size = args.batch_size
    hash_step_size = args.step_size
    if (n_experts := args.n_experts) is None:
        # Infer the number of experts from the model name
        n_experts = int(MODEL.split("-")[-1])

    # Create a queue for hash table, the queue size may need to be tuned for performance
    hash_queue = queue.Queue(maxsize=16)

    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)
    logger.info(f"Experiment Configurations: SODA, Dataset: {DATASET}, Model: {MODEL}, Batch Size: {batch_size}")
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

    # Load model
    model = SwitchTransformersClassificationModel(2, MODEL=MODEL, BASEDIR=BASEDIR, offloading=True)

    # Load dataset
    eval_loader, hash_eval_loader, pre_proc = get_text_cls_loader(
        DATASET,
        MODEL,
        f"{BASEDIR}/tmp",
        split=SPLIT,
        hash_loader=True,
        batch_size=batch_size,
        hash_step_size=hash_step_size,
    )

    # Load pretrained hash predictor
    # TODO: This might needs to be configured based on model and dataset
    #       Currently I only consider model
    # TODO: Check y_keys applies to all models
    predictor = SimpleLSTMClassifierSparseAttention(num_classes=n_experts)
    if os.path.exists(f"{BASEDIR}/data/{DATASET}/{MODEL}/hash_predictor.pt"):
        predictor.load_state_dict(torch.load(f"{BASEDIR}/data/{DATASET}/{MODEL}/hash_predictor.pt")["model_state_dict"])
    else:
        logging.warning("Hash predictor not found, will use randomly initialized hash table.")
    # Move the model to GPU
    model.cpu()

    deamon = SODAManager(model, hash_eval_loader, n_experts, topk=args.topk, predictor_model=predictor)

    with monitor_gputil():
        start.record()
        # Use separate thread to run prediction and try to fill the queue
        deamon_thread = SODAThread(deamon, hash_queue)
        deamon_thread.start()
        move_to_device(model, "cuda:0")
        hash_table, expert_lists = hash_queue.get(block=True)
        deamon.move_experts(expert_lists)

        end.record()
        torch.cuda.synchronize()
        gpu_time = start.elapsed_time(end) / 1000

        start.record()
        total_loading_time = 0
        total_forward_time = 0
        total_samples = 0  # total number of samples
        for step, inputs in enumerate(eval_loader):
            start_move = torch.cuda.Event(enable_timing=True)
            end_move = torch.cuda.Event(enable_timing=True)
            start_step = torch.cuda.Event(enable_timing=True)
            end_step = torch.cuda.Event(enable_timing=True)

            if step > 0 and (step * batch_size) % hash_step_size == 0:
                start_move.record()
                hash_table, expert_lists = hash_queue.get(block=True)
                deamon.move_experts(expert_lists)
                end_move.record()

            start_step.record()
            inputs = pre_proc(inputs)
            end_step.record()
            torch.cuda.synchronize()
            if step > 0 and (step * batch_size) % hash_step_size == 0:
                gpu_time += start_move.elapsed_time(end_move) / 1000.0
            loading_time = start_step.elapsed_time(end_step) / 1000.0
            total_loading_time += loading_time

            batch_size = inputs["input_ids"].size(0)  # assuming your inputs dict contains "input_ids" key

            start_step.record()
            model(
                inputs["input_ids"],
                inputs["attention_mask"],
                hash_table=[
                    (hash_table, ((step * batch_size) % hash_step_size) // batch_size, key, args.topk)
                    for key in list(hash_table.keys())[::-1]
                ],
            )
            end_step.record()
            torch.cuda.synchronize()
            forward_time = start_step.elapsed_time(end_step) / 1000.0
            total_samples += batch_size
            total_forward_time += forward_time
            del inputs
            # Throughput for each forward pass
            throughput_each_forward = batch_size / forward_time
            logger.debug(f"Throughput for step {step}: {throughput_each_forward} samples/second")

            if DATASET == "c4_en" and step > len(eval_loader) / 10:
                break

        deamon_thread.join()
        end.record()
        
    torch.cuda.synchronize()
    end_to_end_time = start.elapsed_time(end) / 1000.0

    # Throughput for the entire process
    total_throughput = total_samples / end_to_end_time
    logger.debug(f"Total throughput: {total_throughput} samples/second")

    gpu_time_portion = (gpu_time / end_to_end_time) * 100
    loading_time_portion = (total_loading_time / end_to_end_time) * 100
    forward_time_portion = (total_forward_time / end_to_end_time) * 100

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
        "gpu_time": gpu_time,
        "gpu_time_portion": gpu_time_portion,
        "loading_time": total_loading_time,
        "loading_time_portion": loading_time_portion,
        "forward_time": total_forward_time,
        "forward_time_portion": forward_time_portion,
        "end_to_end_time": end_to_end_time,
        "average_loading_time_portion": average_loading_time_portion,
        "average_forward_time_portion": average_forward_time_portion,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }

    print(output_dict)

    # Save the result is any of the baseline runs
    logger.debug(output_dict)
    if "output_dict" in locals():
        output_dict["dataset"] = DATASET
        output_dict["model"] = MODEL

        report_name = f"0_latency_id_{args.running_id}.json"
        if os.path.exists(report_name):
            logger.warning(f"Report {report_name} already exists, will be overwritten!")

        with open(os.path.join(REPORT_PATH_BASE, f"0_latency_id_{args.running_id}.json"), "w") as outfile:
            json.dump(output_dict, outfile)
            outfile.write("\n")
    del model, eval_loader, hash_eval_loader, pre_proc
    return output_dict


if __name__ == "__main__":
    options = argparse.ArgumentParser()
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
    options.add_argument("--n_experts", type=int, default=None, help="Number of experts in the hash table")
    options.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    options.add_argument("--step_size", type=int, default=64, help="Step size for hash prediction")
    options.add_argument("--verbose", action="store_true", help="Show debug outputs")
    options.add_argument("--running_id", type=int, default=0, help="refers the id of experiments running")

    # Parse the arguments
    args = options.parse_args()
    main(args)
    # SwitchTransformersSparseMLP(nn.Module) -> input: hidden_states; output: hidden_states, (router_logits, expert_index)
    # SwitchTransformersLayerFF(nn.Module)   -> in-built two kinds of mlps; input: hidden_states; output: output hidden state
    # Only need to change the code in SwitchTransformersBlock, such that it will replace the 'SwitchTransformersLayerFF' by \
    # specific expert given the hasing table

    # Test perplexity:
    # get the special loader, import form other files
    # load the special model, directly load from other files
    # compute the final score
