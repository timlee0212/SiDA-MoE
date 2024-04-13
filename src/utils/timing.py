import torch
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersConfig,
    SwitchTransformersDenseActDense,
    SwitchTransformersLayerCrossAttention,
    SwitchTransformersLayerFF,
    SwitchTransformersLayerNorm,
    SwitchTransformersLayerSelfAttention,
    SwitchTransformersPreTrainedModel,
    SwitchTransformersTop1Router,
)

# Define a wrapper class for SwitchTransformersTop1Router
class TimedSwitchTransformersTop1Router(torch.nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.module = original_module
        self.total_time = 0

    def forward(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Call the original module's forward method
        result = self.module(*args, **kwargs)

        end.record()
        torch.cuda.synchronize()
        self.total_time += start.elapsed_time(end) / 1000.0
        return result

# Function to replace the original modules with their timed versions
def add_timing_hooks(model):
    for name, module in model.named_children():
        if isinstance(module, SwitchTransformersTop1Router):
            setattr(model, name, TimedSwitchTransformersTop1Router(module))
        else:
            add_timing_hooks(module)