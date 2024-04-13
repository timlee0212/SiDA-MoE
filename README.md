# MOE

### Collecting Activations for large models

1. Run `python main.py --model=xxx --sharding`. The script will load the pretrained weight from HF to our customized model and save the weight in a sharded format at ./result/[DATABASE]/[MODEL]/ShardedCkpt
2. Run `python main.py --model=xxx` to perform inference with the HF load_and_dispatch and collect the activations for use.

TODO:

[ ] Add Disk Offload Function.
[ ] Process sharded format when the model size is larger than the main memory.