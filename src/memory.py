import argparse
import logging
import sys

from transformers import modeling_utils

from utils import config

logger = logging.getLogger("mem_test")
BASEDIR = config.BASEDIR
# Disable the warnings related to weight
modeling_utils.logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler(sys.stdout))


import torch

from model import SwitchTransformersClassificationModel
from utils import get_module_memory_usage, measure_layer_memory

# Assuming that we are on a CUDA machine
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    logger.info("CUDA is not available. The code will run on CPU instead.")
    device = torch.device("cpu")


def print_memory_usage_per_module(module, prefix=""):
    logger = logging.getLogger("memory")
    fixed_memory = 0
    expert_memory = 0
    for n, m in module.named_modules():
        if n in [f"switch_transformers.encoder.block.{i}.layer.0" for i in range(12)]:
            fixed_memory += get_module_memory_usage(m)
            logger.debug(
                f"{prefix}{type(module).__name__}: {get_module_memory_usage(m)/ 1024**3:.6f} GB"
            )  # / 1024**3  convert to GB
        elif n in [f"switch_transformers.encoder.block.{i}.layer.1" for i in [0, 2, 4, 6, 8, 10]]:
            fixed_memory += get_module_memory_usage(m)
            logger.debug(
                f"{prefix}{type(module).__name__}: {get_module_memory_usage(m)/ 1024**3:.6f} GB"
            )  # / 1024**3  convert to GB
        elif n in [f"switch_transformers.encoder.block.{i}.layer.1" for i in [1, 3, 5, 7, 9, 11]]:
            expert_memory += get_module_memory_usage(m)
            logger.debug(
                f"{prefix}{type(module).__name__}: {get_module_memory_usage(m)/ 1024**3:.6f} GB"
            )  # / 1024**3  convert to GB
        else:
            logger.debug(n)
    # for name, child in module.named_children():
    #     # if name == "switch_transformers.encoder":
    #         print_memory_usage_per_module(child, prefix=f'{prefix}{name}.')
    return fixed_memory, expert_memory


def measure_all_layers(module, input, prefix=""):
    fixed_memory = 0
    expert_memory = 0
    for n, layer in module.named_modules():
        if n in [f"switch_transformers.encoder.block.{i}.layer.0" for i in range(12)]:
            mem, input = measure_layer_memory(layer, input, device)
            fixed_memory += mem
            logger.debug(f"{n}: {mem / 1024**3:.6f} GB")  # / 1024**3  convert to GB
        elif n in [f"switch_transformers.encoder.block.{i}.layer.1" for i in [0, 2, 4, 6, 8, 10]]:
            mem, input = measure_layer_memory(layer, input, device)
            fixed_memory += mem
            logger.debug(f"{n}: {mem/ 1024**3:.6f} GB")  # / 1024**3  convert to GB
        elif n in [f"switch_transformers.encoder.block.{i}.layer.1" for i in [1, 3, 5, 7, 9, 11]]:
            mem, input = measure_layer_memory(layer, input, device)
            expert_memory += mem
            logger.debug(f"{n}: {mem/ 1024**3:.6f} GB")  # / 1024**3  convert to GB
        # else:
        #     logger.debug(n)
    # for name, child in module.named_children():
    #     # if name == "switch_transformers.encoder":
    #         print_memory_usage_per_module(child, prefix=f'{prefix}{name}.')
    return fixed_memory, expert_memory


def register_hook(num_activated=1):
    def hook_fn(module, input, output):
        # This hook function will alter the output so that to control the sparsity of the activated experts
        output = list(output)
        output[0] = torch.zeros_like(output[0])
        output[0][..., 0:num_activated] = 1
        return output

    return hook_fn


if __name__ == "__main__":
    options = argparse.ArgumentParser()
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
    options.add_argument("--verbose", action="store_true", help="Print the memory usage of each layer")

    if options.parse_args().verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    MODEL = options.parse_args().model
    NUM_EXPERTS = int(MODEL.split("-")[-1])

    # Initialize custom model
    model = SwitchTransformersClassificationModel(2, MODEL, BASEDIR)

    if False:  # measure static
        fixed_memory, expert_memory = print_memory_usage_per_module(model)
        logger.debug(
            f"{type(model).__name__} -- Fixed Mem: {fixed_memory/ 1024**3:.6f} GB,  Expert Mem: {expert_memory/ 1024**3:.6f} GB"
        )  # / 1024**3  convert to GB
    else:
        mem_usage = []

        for n, m in model.named_modules():
            if n.endswith("layer.1.mlp.router"):
                handle = m.register_forward_hook(register_hook(num_activated=1))

        for num_activated in range(1, NUM_EXPERTS + 1):
            handle.remove()

            # add a hook to rounter
            deactivated_experts_list = []
            for j in [1, 3, 5, 7, 9, 11]:
                deactivated_experts_list += [
                    f"switch_transformers.encoder.block.{j}.layer.1.mlp.experts.expert_{i}."
                    for i in range(num_activated, NUM_EXPERTS)
                ]

            for n, m in model.named_modules():
                if n.endswith("layer.1.mlp.router"):
                    handle = m.register_forward_hook(register_hook(num_activated=num_activated))

            # # Create a dummy input
            # input = torch.randn(1, 512, 768).to(device)
            model.to(device)

            # for n, m in model.named_modules():
            #     # remove some of the experts
            #     for name in deactivated_experts_list:
            #         if name in n:
            #             m.to("cpu")

            # fixed_memory, expert_memory = measure_all_layers(model, input)
            # logger.debug(f'{type(model).__name__} -- Fixed Mem: {fixed_memory/ 1024**3:.6f} GB,  Expert Mem: {expert_memory/ 1024**3:.6f} GB') #/ 1024**3  convert to GB
            with torch.no_grad():
                input = torch.zeros(1, 512).to(device)
                torch.cuda.reset_peak_memory_stats(device)
                model(input.long(), None)
                peak_memory = torch.cuda.max_memory_allocated(device)
                logger.debug(
                    f"[{MODEL}] Percentage of activated Experts: {num_activated/NUM_EXPERTS:.2%} Peak Memory: {peak_memory/ 1024**3:.6f} GB"
                )  # / 1024**3  convert to GB
                mem_usage.append((num_activated / NUM_EXPERTS, num_activated, peak_memory / 1024**3))

        torch.save(mem_usage, f"{BASEDIR}/results/{MODEL}_mem_usage.pt")
