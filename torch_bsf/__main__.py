from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow
from mlflow import autolog

from torch_bsf import BezierSimplexDataModule
from torch_bsf.bezier_simplex import BezierSimplex, load, randn
from torch_bsf.validator import index_list, int_or_str, validate_simplex_indices

parser = ArgumentParser(
    prog="python -m torch_bsf", description="Bézier simplex fitting"
)
parser.add_argument("--params", type=Path, required=True)
parser.add_argument("--values", type=Path, required=True)
parser.add_argument("--meshgrid", type=Path)
parser.add_argument("--degree", type=int)
parser.add_argument("--init", type=Path)
parser.add_argument("--fix", type=index_list)
parser.add_argument("--header", type=int, default=0)
parser.add_argument(
    "--normalize", type=str, choices=("none", "max", "std", "quantile"), default="none"
)
parser.add_argument("--split_ratio", type=float, default=1.0)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_epochs", type=int, default=2)
parser.add_argument("--smoothness_weight", type=float, default=0.0)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--strategy", type=str, default="auto")
parser.add_argument("--devices", type=int_or_str, default="auto")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--precision", type=str, default="32-true")
parser.add_argument("--enable_checkpointing", action="store_true")
parser.add_argument("--log_every_n_steps", type=int, default=1)
parser.add_argument(
    "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
)

args = parser.parse_args()

if args.degree is None and args.init is None:
    raise ValueError("Either --degree or --init must be specified")
if args.degree is not None and args.init is not None:
    raise ValueError("Either --degree or --init must be specified, not both")

meshgrid: Path = args.params if (args.meshgrid is None or args.meshgrid.is_dir()) else args.meshgrid

autolog(
    log_input_examples=(args.loglevel >= 2),
    log_model_signatures=False,
    log_models=False,
    disable=(args.loglevel <= 0),
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=(args.loglevel <= 0),
)

dm = BezierSimplexDataModule(
    params=args.params,
    values=args.values,
    header=args.header,
    batch_size=args.batch_size,
    split_ratio=args.split_ratio,
    normalize=args.normalize,
)

if args.init:
    _loaded = load(args.init)
    _same_weight = getattr(_loaded, "smoothness_weight", None) == args.smoothness_weight
    _has_adjacency = hasattr(_loaded, "adjacency_indices_")
    if _same_weight and _has_adjacency:
        bs = _loaded
    else:
        bs = BezierSimplex(_loaded.control_points, smoothness_weight=args.smoothness_weight)
else:
    bs = randn(
        n_params=dm.n_params,
        n_values=dm.n_values,
        degree=args.degree,
        smoothness_weight=args.smoothness_weight,
    )

fix: list[list[int]] = args.fix or []
validate_simplex_indices(fix, bs.n_params, bs.degree)

for index in fix:
    bs.fix_row(index)

trainer = Trainer(
    accelerator=args.accelerator,
    strategy=args.strategy,
    devices=args.devices,
    precision=args.precision,
    num_nodes=args.num_nodes,
    max_epochs=args.max_epochs,
    enable_checkpointing=args.enable_checkpointing,
    log_every_n_steps=args.log_every_n_steps,
    callbacks=[EarlyStopping(monitor="val_mse")],
)
trainer.fit(bs, dm)

if args.loglevel >= 2:
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec
    import mlflow.pytorch

    # Use float64 in signature to match CSV/JSON input (float32/float16 models auto-convert
    # via torch.as_tensor in forward). Shape (-1, n) validates column count without dtype lock-in.
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float64"), (-1, bs.n_params))]),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1, bs.n_values))]),
    )
    last_run = mlflow.last_active_run()
    if mlflow.active_run() is not None or last_run is None:
        mlflow.pytorch.log_model(bs, "model", signature=signature)
    else:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.pytorch.log_model(bs, "model", signature=signature)

# search for filename
fn = f"{args.params.name},{args.values.name},meshgrid,d_{args.degree},r_{args.split_ratio}.csv"

ts = dm.load_data(meshgrid)
xs = bs.forward(ts)
xs = dm.inverse_transform(xs)

# save meshgrid
x = xs.to("cpu").detach().numpy()
np.savetxt(fn, x)
print(f"Meshgrid saved: {fn}")
