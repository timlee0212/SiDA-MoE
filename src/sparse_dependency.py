BASEDIR = "/scratch/xiangyj/moe"

from typing import Dict, List
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch
import math
from torch.optim import Optimizer
from typing import Iterable, Tuple
from torch import nn
import random
import string
import datasets
import copy

from datasets.iterable_dataset import IterableDataset
from torch.utils.data import DataLoader

import os
import json


import yaml

class routerHook:
    def __init__(self, model):
        self.activation_dict = {}
        for name, module in model.named_modules():
            if "encoder" in name:
                if ".mlp.router.classifier" in name:
                    self.activation_dict[name] = {"ptr":0}
                    module.register_forward_hook(self.hook(name))

    def hook(self, name):
        def hook_fn(module, input, output):
            ptr = self.activation_dict[name]["ptr"]
            self.activation_dict[name][ptr] = output.clone().detach().cpu()
            self.activation_dict[name]["ptr"] += 1            
        return hook_fn


def clear_all_hooks(model):
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
            
        # Clear forward pre-hooks
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()
            
        # Clear backward hooks
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()


def random_perturb(input_ids, portion, device, exclude):
    """
    Randomly perturb a portion of the input tensor.
    
    Args:
    - input_ids (torch.Tensor): A tensor of shape [1, 512] containing integer values.
    - portion (float): The fraction of the tensor to perturb (between 0 and 1).
    - exclude (int): The position in the tensor to exclude from perturbation.

    Returns:
    - torch.Tensor: The perturbed tensor.
    """

    # Calculate the number of positions to change
    num_positions = int(input_ids.shape[1] * portion)

    # Generate a list of positions to change, excluding the specified position
    change_positions = torch.randperm(input_ids.shape[1])[:num_positions]
    change_positions = change_positions[change_positions != exclude]

    # Get the integer at the exclude position
    excluded_val = input_ids[0, exclude].item()

    # Generate new random values for the change positions
    new_values = torch.randint(0, 32100, (change_positions.shape[0],)).to(device)
    # Ensure the excluded value is not among the new values
    while excluded_val in new_values:
        new_values = torch.randint(0, 32100, (change_positions.shape[0],)).to(device)
    
    # Assign the new values to the input tensor
    input_ids[0, change_positions] = new_values

    return input_ids

def random_perturb_selected(input_ids, device, exclude, include):
    """
    Randomly perturb a portion of the input tensor.
    
    Args:
    - input_ids (torch.Tensor): A tensor of shape [1, 512] containing integer values.
    - portion (float): The fraction of the tensor to perturb (between 0 and 1).
    - exclude (int): The position in the tensor to exclude from perturbation.

    Returns:
    - torch.Tensor: The perturbed tensor.
    """
    
    import copy
    
    ori = copy.deepcopy(input_ids)
    
    change_positions = torch.tensor(include)

    new_values = torch.randint(0, 32100, (change_positions.shape[0], change_positions.shape[1])).to(device)
    
    input_ids[torch.arange(change_positions.size(0)).unsqueeze(1), change_positions] = new_values

    
    return input_ids


def random_swap(input_ids, portion, device, exclude):
    """
    Randomly select a portion of the input tensor and randomly swap their positions.
    
    Args:
    - input_ids (torch.Tensor): A tensor of shape [1, 512] containing integer values.
    - portion (float): The fraction of the tensor to perturb (between 0 and 1).
    - exclude (int): The position in the tensor to exclude from perturbation.

    Returns:
    - torch.Tensor: The perturbed tensor.
    """
    assert 0 <= portion <= 1, "portion should be between 0 and 1"
    assert 0 <= exclude < input_ids.shape[1], "exclude should be within tensor dimensions"

    # Calculate number of elements to perturb
    total_elements = input_ids.shape[1]
    num_to_perturb = int(total_elements * portion)

    # Randomly select elements to perturb (excluding the 'exclude' index)
    perturb_indices = torch.randperm(total_elements)[:num_to_perturb]
    perturb_indices = perturb_indices[perturb_indices != exclude]

    # Randomly swap the positions of selected elements
    swapped_indices = perturb_indices[torch.randperm(perturb_indices.shape[0])]

    # Create a copy of the input tensor
    perturbed_tensor = input_ids.clone()
    
    # Swap the values
    perturbed_tensor[0, perturb_indices] = input_ids[0, swapped_indices]
    

    return perturbed_tensor

def sparse_dependency(model, exclude_position, device, dependency="token"):
    ratio_list = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    scores_list = []
    with torch.no_grad():
        for i in range(1, 12):
            batch = torch.load(f"./tmp/batch_{i}.dict")
            for j in range(batch['input_ids'].shape[0]):
                scores = {}
                for kn in ratio_list:
                    scores[kn] = 0.0
                    
                sample = {}
                sample['input_ids'], sample['labels'] = batch['input_ids'][j:j+1], batch['labels'][j:j+1]

                router_hook = routerHook(model)
                input_ids  = copy.deepcopy(sample['input_ids'])
                labels     = copy.deepcopy(sample['labels'])
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                # == pick the results from the exclude position ==
                base_activation_dict = {}
                for key in router_hook.activation_dict:
                    base_activation_dict[key] = router_hook.activation_dict[key][0].argmax(dim=-1).view(-1).numpy()[exclude_position]
                    
                clear_all_hooks(model)
                
                for perturb_portion in ratio_list:
                    for perturb_id in range(3):
                        
                        router_hook = routerHook(model)
                        input_ids  = copy.deepcopy(sample['input_ids'])
                        labels     = copy.deepcopy(sample['labels'])
                        input_ids, labels = input_ids.to(device), labels.to(device)                      
                        
                        if dependency == "position":
                            input_ids = random_swap(input_ids, perturb_portion, device, exclude=exclude_position)
                        if dependency == "token"
                            input_ids = random_perturb(input_ids, perturb_portion, device, exclude=exclude_position)

                        outputs = model(input_ids=input_ids, labels=labels)
                        
                        # == pick the results from the exclude position ==
                        activation_dict = {}
                        for key in router_hook.activation_dict:
                            activation_dict[key] = router_hook.activation_dict[key][0].argmax(dim=-1).view(-1).numpy()[exclude_position]

                        clear_all_hooks(model)
                        
                        for k_id, key in enumerate(router_hook.activation_dict.keys()):
                            score = base_activation_dict[key] != activation_dict[key]
                            if k_id == 0 :
                                break 
                        
                        scores[perturb_portion] += score/10
                        
                scores_list.append(scores) # this is a list of dictionary
        
        result = {}
        for kn in ratio_list:
            result[kn] = 0.0        
        # Sum the values from each dictionary
        for s in scores_list:
            for key, value in s.items():
                result[key] += value
        
        # Divide by the number of dictionaries to get the average
        for key in result:
            result[key] /= len(scores_list)
        
        return result

def load_existing_results(filename):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            content = file.read().strip()
            if content:  # Check if file has content
                return json.loads(content)
    return {}

def save_results(filename, results):
    with open(filename, "w") as file:
        json.dump(results, file)

def save_sparse_dependency(model, device, dependency="token"):
    # Define the range for exclude_position. For example, if you want to randomly select from values between 0 and 999:
    exclude_position_range = list(range(512))
    
    # Randomly select 100 exclude_positions
    exclude_positions = random.sample(exclude_position_range, 100)

    results_filename = f"{BASEDIR}/128_{dependency}_dependency.json"

    all_results = load_existing_results(results_filename)
    
    # Run the sparse_dependency function for each exclude_position
    for position in exclude_positions:
        print(">>>>> Check sparsity for position ", position)
        # If this position's result already exists, skip
        if str(position) in all_results:
            continue
        
        result = sparse_dependency(model, position, device)
        all_results[position] = result
        
        # Save all results to a JSON file after each iteration
        save_results(results_filename, all_results)
                            

if __name__ =="__main__":
    BASEDIR = "/scratch/xiangyj/moe"
    device  = "cuda:0"
    import torch
    from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, T5ForConditionalGeneration, SwitchTransformersEncoderModel

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-128", cache_dir=f"{BASEDIR}/tmp/")
    model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-128", cache_dir=f"{BASEDIR}/tmp/")
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    model = model.to(device)
    
    save_sparse_dependency(model, device)

