# pytorch-bsf
![pipy](https://github.com/rafcc/pytorch-bsf/actions/workflows/python-package.yml/badge.svg)
![conda](https://github.com/rafcc/pytorch-bsf/actions/workflows/python-package-conda.yml/badge.svg)
![documents](https://github.com/rafcc/pytorch-bsf/actions/workflows/sphinx-pages.yml/badge.svg)

PyTorch implementation of Bezier simplex fitting.

<img src="./docs/_static/bezier-simplex.png" width="49%"><img src="./docs/_static/bezier-simplex-fitting.png" width="49%">

The Bezier simplex is a high-dimensional generalization of the Bezier curve.
Mathematically, it is a polynomial map from a simplex to a Euclidean space determined by a set of vectors called the control points.
This package provides an algorithm to fit a Bezier simplex to given data points.

See the following papers for technical details.
- Kobayashi, K., Hamada, N., Sannai, A., Tanaka, A., Bannai, K., & Sugiyama, M. (2019). Bézier Simplex Fitting: Describing Pareto Fronts of´ Simplicial Problems with Small Samples in Multi-Objective Optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 2304-2313. https://doi.org/10.1609/aaai.v33i01.33012304
- Tanaka, A., Sannai, A., Kobayashi, K., & Hamada, N. (2020). Asymptotic Risk of Bézier Simplex Fitting. Proceedings of the AAAI Conference on Artificial Intelligence, 34(03), 2416-2424. https://doi.org/10.1609/aaai.v34i03.5622


## Requirements

Python 3.8, 3.9, 3.10.


## Installation-free Execution

Download the latest [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it.
Then, install MLflow on your conda environment:
```
conda install mlflow
```

Run the following command:
```
mlflow run https://github.com/rafcc/pytorch-bsf \
  -P data=data.tsv \
  -P label=label.tsv \
  -P degree=3
```
which automatically sets up the environment and runs an experiment:
1. Download the latest pytorch-bsf into a temporary directory.
2. Create a new conda environment and install dependencies in it.
3. Run an experiment on the temporary directory and environment.

|Parameter|Type|Default|Description|
|-|-|-|-|
|data|path|required|The data file. The file should contain a numerical matrix in the TSV format: each row represents a record that consists of features separated by Tabs or spaces.|
|label|path|required|The label file. The file should contain a numerical matrix in the TSV format: each row represents a record that consists of outcomes separated by Tabs or spaces.|
|degree|int (x >= 1)|required|The degree of the Bezier simplex.|
|header|int (x >= 0)|`0`|The number of header lines in data/label files.|
|delimiter|str|`" "`|The delimiter of values in data/label files.|
|normalize|`"max"`, `"std"`, `"quantile"`|`None`|The data normalization: `max` scales each feature as the minimum is 0 and the maximum is 1, suitable for uniformly distributed data; `std` does as the mean is 0 and the standard deviation is 1, suitable for nonuniformly distributed data; `quantile` does as 5%-quantile is 0 and 95%-quantile is 1, suitable for data containing outliers; `None` does not perform any scaling, suitable for pre-normalized data.|
|split_ratio|float (0.0 < x < 1.0)|`0.5`|The ratio of training data against validation data.|
|batch_size|int (x >= 0)|`0`|The size of minibatch. The default uses all records in a single batch.|
|max_epochs|int (x >= 1)|`1000`|The number of epochs to stop training.|
|accelerator|`"auto"`, `"cpu"`, `"gpu"`, etc.|`"auto"`|Accelerator to use. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/accelerator.html).|
|devices|int (x >= -1)|`None`|The number of accelerators to use. By default, use all available devices. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html).|
|num_nodes|int (x >= 1)|`1`|The number of compute nodes to use. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html).|
|strategy|`"dp"`, `"ddp"`, `"ddp_spawn"`, etc.|`None`|Distributed strategy. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/strategy.html).|
|loglevel|int (0 <= x <= 2)|`2`|What objects to be logged. `0`: nothing; `1`: metrics; `2`: metrics and models.|


## Installation

```
pip install pytorch-bsf
```


## Fitting via CLI

This package provides a command line interface to train a Bezier simplex with a dataset file.

Execute the `torch_bsf` module:
```
python -m torch_bsf \
  --data=data.tsv \
  --label=label.tsv \
  --degree=3
```


## Fitting via Script

Train a model by `fit()`, and call the model to predict.
```python
import torch
import torch_bsf

# Prepare training data
ts = torch.tensor(  # parameters on a simplex
    [
        [3/3, 0/3, 0/3],
        [2/3, 1/3, 0/3],
        [2/3, 0/3, 1/3],
        [1/3, 2/3, 0/3],
        [1/3, 1/3, 1/3],
        [1/3, 0/3, 2/3],
        [0/3, 3/3, 0/3],
        [0/3, 2/3, 1/3],
        [0/3, 1/3, 2/3],
        [0/3, 0/3, 3/3],
    ]
)
xs = 1 - ts * ts  # values corresponding to the parameters

# Train a model
bs = torch_bsf.fit(params=ts, values=xs, degree=3)

# Predict by the trained model
t = [[0.2, 0.3, 0.5]]
x = bs(t)
print(f"{t} -> {x}")
```


## Documents

See documents for more details.
https://rafcc.github.io/pytorch-bsf/


## Author

RIKEN AIP-FUJITSU Collaboration Center (RAFCC)


## License

MIT
