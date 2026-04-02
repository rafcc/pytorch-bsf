from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow import autolog
from pl_crossvalidate import KFoldTrainer

from torch_bsf import BezierSimplexDataModule
from torch_bsf._cli_args import add_common_training_args
from torch_bsf.bezier_simplex import BezierSimplex, load, randn
from torch_bsf.validator import index_list, validate_simplex_indices

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
    "--meshgrid", type=Path, metavar="CSV|DIR",
    help="Path to the meshgrid CSV file used for prediction output; if omitted or set to an existing directory, defaults to --params",
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
    "--freeze", type=index_list, metavar="INDICES",
    help=(
        "JSON-style list-of-lists of simplex indices of control points to freeze during training; "
        "each simplex index list must have length n_params and its entries must sum to the Bezier simplex degree "
        "(either specified by --degree or inferred from the model loaded via --init; "
        "e.g. '[[1,0],[0,1]]' for n_params=2, degree=1)"
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
add_common_training_args(parser)
parser.add_argument(
    "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
)

args = parser.parse_args()

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

fix: list[list[int]] = args.freeze or []
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
