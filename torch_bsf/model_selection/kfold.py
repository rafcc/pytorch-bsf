from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow import autolog
from pl_crossvalidate import KFoldTrainer

from torch_bsf import BezierSimplexDataModule
from torch_bsf.bezier_simplex import BezierSimplex, load, randn
from torch_bsf.validator import index_list, int_or_str, validate_simplex_indices

parser = ArgumentParser(
    prog="python -m torch_bsf.model_selection.kfold",
    description="Bezier simplex fitting with k-fold cross validation",
)
parser.add_argument(
    "--params", type=Path, required=True, metavar="CSV",
    help="Path to the input parameters CSV file",
)
parser.add_argument(
    "--values", type=Path, required=True, metavar="CSV",
    help="Path to the output values CSV file",
)
parser.add_argument(
    "--meshgrid", type=Path, metavar="CSV",
    help="Path to the meshgrid CSV file used for prediction output; if omitted or set to a directory, defaults to --params",
)
degree_init_group = parser.add_mutually_exclusive_group(required=True)
degree_init_group.add_argument(
    "--degree", type=int, metavar="N",
    help="Degree of the Bezier simplex (required when --init is not given)",
)
degree_init_group.add_argument(
    "--init", type=Path, metavar="PT",
    help="Path to a pretrained model file to initialize from (required when --degree is not given)",
)
parser.add_argument(
    "--fix", type=index_list, metavar="INDICES",
    help=(
        "JSON-style list-of-lists of simplex indices of control points to freeze during training; "
        "each simplex index list must have length n_params (e.g. '[[0,0],[1,1]]' for n_params=2)"
    ),
)
parser.add_argument(
    "--header", type=int, default=0, metavar="N",
    help="Number of header rows to skip in CSV files (default: 0)",
)
parser.add_argument(
    "--normalize", type=str, choices=("none", "max", "std", "quantile"), default="none",
    metavar="{none,max,std,quantile}",
    help="Normalization applied to values before training (default: none)",
)
parser.add_argument(
    "--num_folds", type=int, default=5, metavar="K",
    help="Number of folds for k-fold cross-validation (default: 5)",
)
parser.add_argument(
    "--shuffle", action=BooleanOptionalAction, default=True,
    help="Shuffle the dataset before splitting into folds (default: True); use --no-shuffle to disable",
)
parser.add_argument(
    "--stratified", action=BooleanOptionalAction, default=True,
    help="Use stratified k-fold splitting (default: True); use --no-stratified to disable",
)
parser.add_argument(
    "--split_ratio", type=float, default=1.0, metavar="R",
    help="Fraction of data used for training; the remainder becomes the validation set (default: 1.0)",
)
parser.add_argument(
    "--batch_size", type=int, metavar="N",
    help="Mini-batch size for training; omit to use full-batch loading",
)
parser.add_argument(
    "--max_epochs", type=int, default=2, metavar="N",
    help="Maximum number of training epochs (default: 2)",
)
parser.add_argument(
    "--smoothness_weight", type=float, default=0.0, metavar="W",
    help="Weight of the smoothness regularization term (default: 0.0)",
)
parser.add_argument(
    "--accelerator", type=str, default="auto", metavar="TYPE",
    help="Hardware accelerator for the Lightning trainer, e.g., 'cpu', 'gpu', 'tpu', 'mps', or 'auto' (default: auto)",
)
parser.add_argument(
    "--strategy", type=str, default="auto", metavar="NAME",
    help="Distributed training strategy for the Lightning trainer, e.g., 'ddp', 'fsdp', 'deepspeed', or 'auto' (default: auto)",
)
parser.add_argument(
    "--devices", type=int_or_str, default="auto", metavar="N|auto",
    help="Number of devices to use, or 'auto' to let the Lightning trainer decide (default: auto)",
)
parser.add_argument(
    "--num_nodes", type=int, default=1, metavar="N",
    help="Number of compute nodes for distributed training (default: 1)",
)
parser.add_argument(
    "--precision", type=str, default="32-true", metavar="PRECISION",
    help="Floating-point precision for the Lightning trainer: '32-true', '16-mixed', 'bf16-mixed', etc. (default: 32-true)",
)
parser.add_argument(
    "--enable_checkpointing", action="store_true",
    help="Enable Lightning model checkpointing during training (disabled by default)",
)
parser.add_argument(
    "--log_every_n_steps", type=int, default=1, metavar="N",
    help="Log training metrics every N optimizer steps (default: 1)",
)
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

x = xs.mean(dim=0)  # mean over folds
x = dm.inverse_transform(x).to("cpu").detach().numpy()
fn = f"{args.params.name},{args.values.name},{args.num_folds}fold,meshgrid,d_{args.degree},r_{args.split_ratio}.csv"
np.savetxt(fn, x)

print(f"Meshgrid saved: {fn}")
