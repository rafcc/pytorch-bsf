from argparse import ArgumentParser
from pathlib import Path

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
parser.add_argument(
    "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
)

args = parser.parse_args()

if args.degree is None and args.init is None:
    raise ValueError("Either --degree or --init must be specified")
if args.degree is not None and args.init is not None:
    raise ValueError("Either --degree or --init must be specified, not both")

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
)

# Returns a dict of stats over the different splits
cross_val_stats = trainer.cross_validate(bs, datamodule=dm)
print(f"cross_val_stats={cross_val_stats}")

# Additionally, we can construct an ensemble from the K trained models
ensemble_model = trainer.create_ensemble(bs)

# search for filename
fn_tmpl = (
    f"{args.params.name},{args.values.name},{args.num_folds}fold,meshgrid,d_{args.degree},r_{args.split_ratio},"
    + "{}.csv"
)
for i in range(1000000):
    fn = Path(fn_tmpl.format(i))
    if not fn.exists():
        break
else:
    raise FileExistsError(fn)

meshgrids = ensemble_model.meshgrid()
print("===========MESHGRIDS===========")
print(meshgrids)
print("===========MESHGRIDS===========")
ts = sum(tx[0] for tx in meshgrids) / args.num_folds
xs = sum(tx[1] for tx in meshgrids) / args.num_folds

# save meshgrid
with open(fn, "w") as f:
    for t, x in zip(ts, xs):
        t = ", ".join(str(v) for v in t.tolist())
        x = ", ".join(str(v) for v in x.tolist())
        f.write(f"{t}, {x}\n")

print(f"Meshgrid saved: {fn}")
