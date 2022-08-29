import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from mlflow import autolog
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch_bsf import BezierSimplex, BezierSimplexDataModule


parser = ArgumentParser(prog="python -m torch_bsf", description="Bezier simplex fitting")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--degree", type=int, required=True)
parser.add_argument("--header", type=int, default=0)
parser.add_argument("--delimiter", type=str)
parser.add_argument("--normalize", type=str, default="none")
parser.add_argument("--split_ratio", type=float, default=0.5)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--accelerator", type=str)
parser.add_argument("--devices", type=int)
parser.add_argument("--num_nodes", type=int)
parser.add_argument("--strategy", type=str)
parser.add_argument("--loglevel", type=int, default=2)  # 0: nothing, 1: metrics, 2: metrics & models
args = parser.parse_args()

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
    data=args.data,
    label=args.label,
    header=args.header,
    delimiter=args.delimiter,
    batch_size=args.batch_size,
    split_ratio=args.split_ratio,
    normalize=args.normalize,
)
bs = BezierSimplex(
    n_params=dm.n_params,
    n_values=dm.n_values,
    degree=args.degree,
)

trainer = pl.Trainer(
    accelerator=args.accelerator,
    devices=args.devices,
    auto_select_gpus=True,
    strategy=args.strategy,
    num_nodes=args.num_nodes,
    max_epochs=args.max_epochs,
    callbacks=[EarlyStopping(monitor="val_mse")],
)
trainer.fit(bs, dm)

# search for filename
fn_tmpl = args.data + f",meshgrid,d_{args.degree},r_{args.split_ratio}," + "{}.csv"
for i in range(1000000):
    fn = fn_tmpl.format(i)
    if not os.path.exists(fn):
        break
else:
    raise FileExistsError(fn)

ts, xs = bs.meshgrid()

# save meshgrid
with open(fn, "w") as f:
    for t, x in zip(ts, xs):
        t = ", ".join(str(v) for v in t.tolist())
        x = ", ".join(str(v) for v in x.tolist())
        f.write(f"{t}, {x}\n")

print(f"Meshgrid saved: {fn}")
