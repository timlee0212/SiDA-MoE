import torch


class routerHook:
    def __init__(self, model, save_dir=None, split="validation"):
        self.activation_dict = {}
        self.name_ptr = 0
        self.dir = save_dir
        self.name_list = []
        self.split = split
        self.handles = []
        for name, module in model.named_modules():
            if ".mlp.router.classifier" in name:
                self.name_list.append(name)
                self.activation_dict[name] = {"ptr": 0}
                handle = module.register_forward_hook(self.hook(name))
                self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def hook(self, name):
        def hook_fn(module, input, output):
            ptr = self.activation_dict[name]["ptr"]
            self.activation_dict[name][ptr] = output.clone().detach().cpu()
            self.activation_dict[name]["ptr"] += 1
            if self.activation_dict[name]["ptr"] % 100 == 0:
                if not self.dir is None:
                    print(f"Save to {self.dir}/activation_{self.split}_large-{self.name_ptr}.pt")
                    torch.save(self.activation_dict, f"{self.dir}/activation_{self.split}_large-{self.name_ptr}.pt")
                self.name_ptr += 1
                ptr = self.activation_dict[name]["ptr"]
                del self.activation_dict

                self.activation_dict = {}
                for n in self.name_list:
                    if ".mlp.router.classifier" in n:
                        self.activation_dict[n] = {"ptr": ptr}

        return hook_fn


class inputHook:
    def __init__(self, model, save_dir, split="validation"):
        self.data_dict = {}
        self.name_ptr = 0
        self.dir = save_dir
        self.split = split
        # Save for unregistering the hook
        self.handles = []
        for name, module in model.named_modules():
            if name in ["switch_transformers.encoder.block.0.layer.0", "switch_transformers.encoder.embed_tokens"]:
                self.data_dict[name] = {"ptr": 0}
                handle = module.register_forward_hook(self.hook(name))
                self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def hook(self, name):
        def hook_fn(module, input, output):
            ptr = self.data_dict[name]["ptr"]
            self.data_dict[name][ptr] = input[0].clone().detach().cpu()
            self.data_dict[name]["ptr"] += 1
            if self.data_dict[name]["ptr"] % 100 == 0:
                print(f"Save to {self.dir}/data_{self.split}_large-{self.name_ptr}.pt")
                torch.save(self.data_dict, f"{self.dir}/data_{self.split}_large-{self.name_ptr}.pt")
                self.name_ptr += 1
                ptr = self.data_dict[name]["ptr"]
                del self.data_dict
                self.data_dict = {}

                self.data_dict["switch_transformers.encoder.block.0.layer.0"] = {"ptr": ptr}
                self.data_dict["switch_transformers.encoder.embed_tokens"] = {"ptr": ptr}

        return hook_fn