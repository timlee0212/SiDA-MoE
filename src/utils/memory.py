

import torch


def measure_layer_memory(layer, input, device):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    flag = False
    
    with torch.no_grad():
        try:
            output = layer(input)
        except:
            flag = True
            output = layer(input[0], output_router_logits=False)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(device)
    del output
    torch.cuda.empty_cache()
    
    if flag:
        return (peak_memory, layer(input[0], output_router_logits=False))
    else:
        return (peak_memory, layer(input))

def get_module_memory_usage(module):
    total_memory = 0
    for param in module.parameters(recurse=True):
        total_memory += param.nelement() * param.element_size()
    for buffer in module.buffers(recurse=True):
        total_memory += buffer.nelement() * buffer.element_size()
    return total_memory


