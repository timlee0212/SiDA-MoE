import os
import sys
import pickle
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/"))
import torch

from dataset import get_metrics, get_text_cls_loader
from finetune import get_num_labels
from model import SimpleLSTMClassifierSparseAttention, SwitchTransformersClassificationModel
from model.predictor import build_sparse_rnn, load_raw_data
from utils import SODAManager, config, move_to_device, monitor_gputil
from utils.hooks import inputHook, routerHook
import numpy as np

from pymoo.core.problem import ElementwiseProblem

BASEDIR = config.BASEDIR


@dataclass
class ExpArg:
    topk: int = 3
    model: str = "switch-base-8"
    dataset: str = "sst2"
    n_experts: int = 8
    # CHANGE ME: confirm before run script
    batch_size: int = 1
    eval_bs: int = 64
    step_size: int = 64
    verbose: bool = False
    KD: bool = True


ARGS = ExpArg()

ft_ckpt = os.path.join(config.ckpt_dir, f"{ ARGS.dataset}/{ ARGS.model}/finetuned")
if not os.path.exists(ft_ckpt):
    FileNotFoundError(f"Checkpoint {ft_ckpt} not found. Please run finetuning first.")


class DegradTuningProblem(ElementwiseProblem):
    def __init__(self, n_var=3, **kwargs):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=1, n_constr=n_var, xl=xl, xu=xu, type_var=float, **kwargs)

    # Load dataset; Warp this as function since we need this for several times

    def prepare_data(self, soda=False, split="validation"):
        eval_loader, soda_eval_loader, pre_proc = get_text_cls_loader(
            ARGS.dataset,
            ARGS.model,
            f"{BASEDIR}/tmp",
            batch_size=ARGS.eval_bs,
            split=split,
            hash_loader=soda,
            hash_step_size=ARGS.eval_bs,
        )
        metrics = get_metrics(ARGS.dataset)
        return eval_loader, pre_proc, metrics, soda_eval_loader

    def _evaluate(self, x, out, *args, **kwargs):
        # Load finetuned checkpoint
        out["G"] = [sum(x) - 1.0]
        if len(x) > 1:
            for i in range(1, len(x)):
                out["G"].append(x[i - 1] - x[i])
        model = SwitchTransformersClassificationModel.from_checkpoint(
            get_num_labels(ARGS.dataset),
            MODEL=ARGS.model,
            BASEDIR=BASEDIR,
            ckpt_path=os.path.join(ft_ckpt, "pytorch_model.bin"),
            offloading=True,
            degrad_factor=x,
        ).to("cuda")
        model.eval()
        eval_loader, pre_proc, metrics, soda_eval_loader = self.prepare_data(soda=True)
        predictor = SimpleLSTMClassifierSparseAttention(num_classes=ARGS.n_experts)
        predictor.load_state_dict(torch.load(f"{ft_ckpt}/hash_predictor.pt")["model_state_dict"])
        manager = SODAManager(model, soda_eval_loader, ARGS.n_experts, topk=ARGS.topk, predictor_model=predictor)
        move_to_device(model, "cuda:0")
        for _, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating SODA model"):
            hash_table, experts_list = manager.gen_hash(soft_target=ARGS.KD)
            manager.move_experts(experts_list)
            batch = pre_proc(batch)
            with torch.no_grad():
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    hash_table=[(hash_table, 0, key, ARGS.topk) for key in list(hash_table.keys())[::-1]],
                )
                if ARGS.dataset != "stsb":
                    predictions = torch.argmax(outputs, axis=1)
                else:
                    predictions = predictions[:, 0]
                metrics.add_batch(predictions=predictions, references=batch["label"])
        soda_metrics = metrics.compute()
        out["F"] = [
            -(soda_metrics["accuracy"]),
        ]


from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from multiprocessing.pool import ThreadPool

n_process = 4
pool = ThreadPool(n_process)
runner = StarmapParallelization(pool.starmap)

algorithm = NSGA2(
    pop_size=20,
    n_offsprings=5,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)
problem = DegradTuningProblem(n_var=ARGS.topk, elementwise_runner=runner)
termination = get_termination("n_gen", 40)
res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)
print(res.X, res.F)
X = res.X
F = res.F
pool.close()
