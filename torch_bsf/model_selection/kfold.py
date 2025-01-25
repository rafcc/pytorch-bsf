from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow import autolog
from pl_crossvalidate import KFoldTrainer

from torch_bsf import BezierSimplexDataModule
from torch_bsf.bezier_simplex import load, randn
from torch_bsf.validator import index_list, int_or_str, validate_simplex_indices

parser = ArgumentParser(
    prog="python -m torch_bsf.model_selection.kfold",
    description="Bezier simplex fitting with k-fold cross validation",
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
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--stratified", type=bool, default=True)
parser.add_argument("--split_ratio", type=float, default=1.0)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_epochs", type=int, default=2)
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

meshgrid: Path = args.meshgrid or args.params

autolog(
    log_input_examples=(args.loglevel >= 2),
    log_model_signatures=(args.loglevel >= 2),
    log_models=(args.loglevel >= 2),
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

bs = (
    load(args.init)
    if args.init
    else randn(
        n_params=dm.n_params,
        n_values=dm.n_values,
        degree=args.degree,
    )
)

fix: list[list[int]] = args.fix or []
validate_simplex_indices(fix, bs.n_params, bs.degree)

for index in fix:
    bs.control_points[index].requires_grad = False

trainer = KFoldTrainer(
    num_folds=args.num_folds,
    shuffle=args.shuffle,
    stratified=args.stratified,
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

# Returns a dict of stats over the different splits
cross_val_stats: list[list[dict[str, float]]] = trainer.cross_validate(bs, datamodule=dm)
print(f"{cross_val_stats=}")

# Additionally, we can construct an ensemble from the K trained models
ensemble_model = trainer.create_ensemble(bs)

ts = dm.load_data(meshgrid)
xs = ensemble_model.forward(ts)  # forward for each fold
for k in range(args.num_folds):
    x = xs[k]
    x = dm.inverse_transform(x).to("cpu").detach().numpy()
    fn = f"{args.params.name},{args.values.name},{args.num_folds}fold,meshgrid,d_{args.degree},r_{args.split_ratio},{k}.csv"
    np.savetxt(fn, x)

x = xs.mean(dim=0)  ## mean over folds
x = dm.inverse_transform(x).to("cpu").detach().numpy()
fn = f"{args.params.name},{args.values.name},{args.num_folds}fold,meshgrid,d_{args.degree},r_{args.split_ratio}.csv"
np.savetxt(fn, x)

print(f"Meshgrid saved: {fn}")
