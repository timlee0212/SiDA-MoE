import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

logger = logging.getLogger("Hash_Predictor")
logger.addHandler(logging.StreamHandler(sys.stdout))


def student_loss(loss, s_logit, t_logit, return_t_logits=False, topK=None):
    """Kl/ L1 Loss for student"""
    print_logits = False

    if topK is not None:
        # Get the indices of the top-K largest values in t_logit
        _, topk_indices = torch.topk(t_logit, topK, dim=1)
        # Select only the corresponding elements in s_logit and t_logit
        s_logit = torch.gather(s_logit, -1, topk_indices)
        t_logit = torch.gather(t_logit, -1, topk_indices)

    if loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif loss == "l1softmax":
        loss_fn = F.l1_loss
        s_logit = F.softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach())
    elif loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


class SequenceDataset_c4_en(Dataset):
    def __init__(self, data_dir, y_key="switch_transformers.encoder.block.1.layer.1.mlp.router.classifier"):
        self.data_files = []
        self.label_files = []
        self.y_key = y_key
        self.cache = {}  # In-memory cache

        # Collect data and label files
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if filename.startswith("data_validation_large-") and os.path.isfile(filepath):
                self.data_files.append(filepath)
            elif filename.startswith("activation_validation_large-") and os.path.isfile(filepath):
                self.label_files.append(filepath)

        # Sort both lists using the extracted number
        self.data_files.sort(key=lambda s: int(s.split("-")[-1].split(".")[0]))
        self.label_files.sort(key=lambda s: int(s.split("-")[-1].split(".")[0]))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        labels_dict = []
        data = torch.load(self.data_files[idx])  # Memory-mapped loading
        data = data["switch_transformers.encoder.block.0.layer.0"]
        labels = torch.load(self.label_files[idx])  # Memory-mapped loading

        common_keys = set(data.keys()) & set(labels[self.y_key].keys())
        if "ptr" in common_keys:
            common_keys.remove("ptr")

        keys_list = list(common_keys)

        data = torch.cat([data[key] for key in keys_list[1:]], dim=0)

        for key in labels.keys():
            labels_layer = labels[key]
            labels_layer = torch.cat([labels_layer[key_idx] for key_idx in keys_list[1:]], dim=0)
            labels_layer = torch.argmax(labels_layer, dim=-1)
            labels_dict.append(labels_layer.unsqueeze(0))

        return data, torch.cat(labels_dict, dim=0)


class SequenceDataset(Dataset):
    def __init__(self, data_dir, soft_target=False, split="validation"):
        self.data_files = []
        self.label_files = []
        self.cache = {}  # In-memory cache
        self.soft_target = soft_target

        # Collect data and label files
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if filename.startswith(f"data_{split}_large-") and os.path.isfile(filepath):
                self.data_files.append(filepath)
            elif filename.startswith(f"activation_{split}_large-") and os.path.isfile(filepath):
                self.label_files.append(filepath)
                # Sort both lists using the extracted number
        self.data_files.sort(key=lambda s: int(s.split("-")[-1].split(".")[0]))
        self.label_files.sort(key=lambda s: int(s.split("-")[-1].split(".")[0]))

        labels_dict = []
        data = [torch.load(path) for path in self.data_files]  # Memory-mapped loading
        merged_data = {}
        for data_portion in data:
            merged_data.update(data_portion["switch_transformers.encoder.block.0.layer.0"])
        self.data = merged_data
        del data

        labels = [torch.load(path) for path in self.label_files]  # Memory-mapped loading
        merged_labels = {key: {} for key in labels[0].keys()}
        for key in labels[0].keys():
            for labels_portion in labels:
                merged_labels[key].update(labels_portion[key])
        self.labels = merged_labels
        del labels

        common_keys = set(self.data.keys()) & set(self.labels[list(self.labels.keys())[0]].keys())
        if "ptr" in common_keys:
            common_keys.remove("ptr")
        self.keys_list = list(common_keys)
        self.keys_list = [self.keys_list[i : i + 1] for i in range(0, len(self.keys_list), 1)]

    def __len__(self):
        return len(self.keys_list)

    def __getitem__(self, idx):
        # Add to cache
        labels_dict = []
        data = torch.cat([self.data[key_idx] for key_idx in self.keys_list[idx]], dim=0)
        for key in self.labels.keys():
            labels_layer = self.labels[key]
            labels_layer = torch.cat([labels_layer[key] for key in self.keys_list[idx]], dim=0)
            if not self.soft_target:  # No KD
                labels_layer = torch.argmax(labels_layer, dim=-1)
            labels_dict.append(labels_layer.unsqueeze(0))

        return data, torch.cat(labels_dict, dim=0)


def load_raw_data(data_dir, soft_target=False):
    train_dataset = SequenceDataset(data_dir, soft_target=soft_target, split="train")
    test_dataset = SequenceDataset(data_dir, soft_target=soft_target, split="validation")

    # Create dataloaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def load_raw_data_c4_en(data_dir):
    dataset = SequenceDataset_c4_en(data_dir)
    train_ratio = 0.8
    # Split the dataset into training and testing subsets
    train_len = int(train_ratio * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # Create dataloaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


# Function to compute top-K accuracy using slices
def _get_accuracy_sliced(model, dataloader, topK=1, KD=False):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    softmax_diff = 0
    baseline_softmax_diff = 0
    total = 0
    slice_size = 4096
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            try:
                batch_data, batch_labels = batch_data.squeeze(0).to("cuda:0"), torch.cat(batch_labels, dim=1).squeeze(
                    0
                ).to("cuda:0")
            except:
                batch_data, batch_labels = batch_data.squeeze(0).to("cuda:0"), batch_labels.squeeze(0).to("cuda:0")
            outputs = []
            for i in range(0, batch_data.size(0), slice_size):
                slice_data = batch_data[i : i + slice_size]
                slice_labels = batch_labels[:, i : i + slice_size]
                slice_output = model(slice_data)

                if KD:
                    num_experts = slice_output.shape[-1]
                    true_idx = torch.argmax(slice_labels.reshape(-1, num_experts), dim=-1)

                    softmax_diff += sum(
                        abs(
                            torch.softmax(slice_output.reshape(-1, num_experts), dim=-1)[
                                torch.arange(true_idx.size(0)), true_idx
                            ]
                            - torch.softmax(slice_labels.reshape(-1, num_experts), dim=-1)[
                                torch.arange(true_idx.size(0)), true_idx
                            ]
                        )
                    )

                    baseline_softmax_diff += sum(
                        abs(
                            1 / 128
                            - torch.softmax(slice_labels.reshape(-1, num_experts), dim=-1)[
                                torch.arange(true_idx.size(0)), true_idx
                            ]
                        )
                    )

                    # Compute Top-K accuracy
                    _, topk_indices = torch.topk(slice_output, topK, dim=-1)
                    topk_indices = topk_indices.reshape(-1, topK).t()
                    correct_tensor = topk_indices.eq(
                        torch.argmax(slice_labels, dim=-1).reshape(1, -1).expand_as(topk_indices)
                    )
                    total += slice_labels.reshape(-1, num_experts).size(0)
                else:
                    # Compute Top-K accuracy
                    _, topk_indices = torch.topk(slice_output, topK, dim=-1)
                    topk_indices = topk_indices.reshape(-1, topK).t()
                    correct_tensor = topk_indices.eq(
                        torch.argmax(slice_labels, dim=-1).reshape(1, -1).expand_as(topk_indices)
                    )
                    total += slice_labels.reshape(-1).size(0)

                correct_k = correct_tensor[:topK].reshape(-1).float().sum(0, keepdim=True).item()
                correct += correct_k

                del slice_data, slice_labels
            del batch_data, batch_labels

    return 100 * correct / total, softmax_diff / total, baseline_softmax_diff / total


# Hash Table prediction class
class SimpleLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=128,
        num_layers=2,
        y_keys=[
            "switch_transformers-encoder-block-1-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-3-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-5-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-7-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-9-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-11-layer-1-mlp-router-classifier",
        ],
    ):
        super(SimpleLSTMClassifier, self).__init__()
        self.compression_fc = nn.Linear(input_dim, hidden_dim)
        self.residual_fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.y_keys = y_keys
        self.fc = nn.ModuleDict({key: nn.Linear(hidden_dim, num_classes) for key in y_keys})

    def to(self, device):
        super(SimpleLSTMClassifier, self).to(device)
        for key in self.y_keys:
            self.fc[key].to(device)
        return self

    def forward(self, x):
        x = self.compression_fc(x)
        x = self.relu(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # x = 0.5*(lstm_out + x)
        # lstm_out, (h_n, c_n) = self.lstm2(x)
        lstm_out = lstm_out + self.residual_fc(x)
        out = [self.fc[key](lstm_out).unsqueeze(0) for key in self.y_keys]  # -1 is used to select the last time step
        return torch.cat(out, dim=0)

    def evaluate(self, eval_loader, topK=1, KD=False):
        return _get_accuracy_sliced(self, eval_loader, topK=topK, KD=KD)

    def save(self, path):
        torch.save({"model_state_dict": self.state_dict()}, path)


class Sparsemax(torch.nn.Module):
    def forward(self, input):
        dim = input.dim() - 1
        sorted_input, _ = torch.sort(input, dim=dim, descending=True)
        cumulative_values = sorted_input.cumsum(dim) - 1
        range_values = torch.arange(1, input.size(dim) + 1, device=input.device).view(1, -1)
        valid_entries = (sorted_input - cumulative_values / range_values) > 0
        rho = valid_entries.sum(dim=dim, keepdim=True)
        tau = (cumulative_values.gather(dim, rho - 1) - 1) / rho.float()
        return torch.max(torch.zeros_like(input), input - tau)


class Attention(torch.nn.Module):
    def __init__(self, method="sparsemax"):
        super(Attention, self).__init__()

        if method == "sparsemax":
            self.activation = Sparsemax()
        else:
            # Implement other methods like Entmax here
            raise ValueError(f"Unknown method: {method}")

    def forward(self, query, key, value):
        # Updated assumption: query is [batch, seq_len, hidden_size], key is [batch, seq_len, hidden_size]
        attention_logits = torch.bmm(query, key.transpose(1, 2))  # [batch, seq_len_query, seq_len_key]

        attention_weights = self.activation(attention_logits)  # Use Sparsemax here
        attention_weights = F.normalize(attention_weights, p=1, dim=2)  # Normalize over the key sequence

        context = torch.bmm(attention_weights, value)  # [batch, seq_len_query, hidden_size]
        return context, attention_weights


# Hash Table prediction class with sparse attention
class SimpleLSTMClassifierSparseAttention(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=128,
        num_layers=2,
        y_keys=[
            "switch_transformers-encoder-block-1-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-3-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-5-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-7-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-9-layer-1-mlp-router-classifier",
            "switch_transformers-encoder-block-11-layer-1-mlp-router-classifier",
        ],
    ):
        super(SimpleLSTMClassifierSparseAttention, self).__init__()

        self.compression_fc = nn.Linear(input_dim, hidden_dim)
        self.residual_fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(method="sparsemax")

        self.y_keys = y_keys
        self.fc = nn.ModuleDict({key: nn.Linear(hidden_dim, num_classes) for key in y_keys})

    def forward(self, x):
        x = self.compression_fc(x)
        x = self.relu(x)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention mechanism
        # Assuming lstm_out acts as both the query and the key
        context, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Add residuals
        context = context + self.residual_fc(x)

        out = [self.fc[key](context).unsqueeze(0) for key in self.y_keys]
        return torch.cat(out, dim=0)

    def save(self, path):
        torch.save({"model_state_dict": self.state_dict()}, path)

    def evaluate(self, eval_loader, topK=1, KD=False):
        return _get_accuracy_sliced(self, eval_loader, topK=topK, KD=KD)

    def to(self, device):
        super(SimpleLSTMClassifierSparseAttention, self).to(device)
        for key in self.y_keys:
            self.fc[key].to(device)
        return self

def build_sparse_rnn(train_dataloader, test_dataloader, num_experts, lr=5e-5, num_epochs=15, test_topk=3, KD=True):
    # Hyperparameters
    slice_size = 2048

    # Initialize model, criterion, and optimizer
    model = SimpleLSTMClassifierSparseAttention(num_classes=num_experts).to("cuda:0")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training loop with slicing
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_data, batch_labels in train_dataloader:
            try:
                batch_data, batch_labels = batch_data.squeeze(0).to("cuda:0"), torch.cat(batch_labels, dim=1).squeeze(
                    0
                ).to("cuda:0")
            except:
                batch_data, batch_labels = batch_data.squeeze(0).to("cuda:0"), batch_labels.squeeze(0).to("cuda:0")

            for i in range(0, batch_data.size(0), slice_size):
                optimizer.zero_grad()  # Zero the parameter gradients
                slice_data = batch_data[i : i + slice_size]
                slice_labels = batch_labels[:, i : i + slice_size]

                slice_output = model(slice_data)
                if not KD:
                    loss = criterion(slice_output.reshape(-1, num_experts), slice_labels.reshape(-1))
                else:
                    loss_CE = criterion(
                        slice_output.reshape(-1, num_experts), torch.argmax(slice_labels, dim=-1).reshape(-1)
                    )

                    loss_l1 = student_loss(
                        "l1softmax",
                        slice_output.reshape(-1, num_experts),
                        slice_labels.reshape(-1, num_experts),
                        topK=min(30, num_experts),
                    )
                    loss = 0.005 * loss_CE + loss_l1

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                del slice_data, slice_labels
            del batch_data, batch_labels
        train_accuracy, softmax_diff, baseline_softmax_diff = _get_accuracy_sliced(
            model, train_dataloader, topK=test_topk, KD=KD
        )
        test_accuracy, softmax_diff, baseline_softmax_diff = _get_accuracy_sliced(
            model, test_dataloader, topK=test_topk, KD=KD
        )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Top: {test_topk}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, SoftMax Difference: {softmax_diff.item():.4f}, Baseline SoftMax Difference: {baseline_softmax_diff.item():.4f}"
        )

    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    # }
    # torch.save(checkpoint, f"{ft_ckpt}/hash_predictor.pt")
    return model, num_epochs


if __name__ == "__main__":
    print(">>>>  Load data")
    ft_ckpt = "/scratch/xiangyj/moe/data/multirc/switch-base-128-old"
    test_topk = 1 
    train_dataloader, test_dataloader = load_raw_data(data_dir=f"{ft_ckpt}/", soft_target=True)
    
    print(">>>> Train sparse LSTM")
    model, _ = build_sparse_rnn(train_dataloader, test_dataloader, num_experts=128, lr=5e-5, num_epochs=100, test_topk=3, KD=True)
    model.save(f"{ft_ckpt}/hash_predictor-100.pt")
    
    print(">>>> Test trained predictor")
    model = SimpleLSTMClassifierSparseAttention(num_classes=128).to("cuda:0")
    model.load_state_dict(torch.load(f"{ft_ckpt}/hash_predictor.pt")["model_state_dict"])
    train_accuracy, softmax_diff, baseline_softmax_diff = _get_accuracy_sliced(model, train_dataloader, topK=test_topk, KD=True)
    test_accuracy, softmax_diff, baseline_softmax_diff = _get_accuracy_sliced(model, test_dataloader, topK=test_topk, KD=True)
    print(f"Epoch [Evaluation], Top: {test_topk}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, SoftMax Difference: {softmax_diff.item():.4f}, Baseline SoftMax Difference: {baseline_softmax_diff.item():.4f}")
