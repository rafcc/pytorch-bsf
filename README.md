# pytorch-bsf
![](./docs/_static/bezier-simplex.png)
![](./docs/_static/bezier-simplex-fitting.png)

PyTorch implementation of Bezier simplex fitting.

The Bezier simplex is a high-dimensional generalization of the Bezier curve.
Mathematically, it is a polynomial map from a simplex to a Euclidean space determined by a set of vectors called the control points.
This package provides an algorithm to fit a Bezier simplex to given data points.

See the following papers for technical details.
- Kobayashi, K., Hamada, N., Sannai, A., Tanaka, A., Bannai, K., & Sugiyama, M. (2019). Bézier Simplex Fitting: Describing Pareto Fronts of´ Simplicial Problems with Small Samples in Multi-Objective Optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 2304-2313. https://doi.org/10.1609/aaai.v33i01.33012304
- Tanaka, A., Sannai, A., Kobayashi, K., & Hamada, N. (2020). Asymptotic Risk of Bézier Simplex Fitting. Proceedings of the AAAI Conference on Artificial Intelligence, 34(03), 2416-2424. https://doi.org/10.1609/aaai.v34i03.5622

## Requirements

Python 3.8 or above.


## How to use

Install the package:
```
$ pip install git+https://github.com/rafcc/pytorch-bsf
```

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
bs = torch_bsf.fit(params=ts, values=xs, degree=3, max_epochs=100)

# Predict by the trained model
t = [[0.2, 0.3, 0.5]]
x = bs(t)
print(f"{t} -> {x}")
```


## Command line interface

This package provides a command line interface to train a Bezier simplex with a dataset file.


### Run as a Python module

Execute the `torch_bsf` module:
```
$ python -m torch_bsf \
  --data=data.tsv \
  --label=label.tsv \
  --degree=3 \
  --header=0 \
  --delimiter=' ' \
  --normalize=none \
  --split_ratio=0.5 \
  --batch_size=32 \
  --max_epochs=1000 \
  --gpus=1 \
  --num_nodes=1 \
  --accelerator=ddp \
  --loglevel=2
```

|Parameter|Type|Default|Description|
|-|-|-|-|
|data|path|required|The data file. The file should contain a numerical matrix in the TSV format: each row represents a record that consists of features separated by Tabs or spaces.|
|label|path|required|The label file. The file should contain a numerical matrix in the TSV format: each row represents a record that consists of outcomes separated by Tabs or spaces.|
|degree|int (x >= 1)|required|The degree of the Bezier simplex.|
|header|int (x >= 0)|`0`|The number of header lines in data/label files.|
|delimiter|str|` `|The delimiter of values in data/label files.|
|normalize|`max`, `std`, `quantile`, or `none`|`none`|The data normalization: `max` scales each feature as the minimum is 0 and the maximum is 1, suitable for uniformly distributed data; `std` does as the mean is 0 and the standard deviation is 1, suitable for nonuniformly distributed data; `quantile` does as 5%-quantile is 0 and 95%-quantile is 1, suitable for data containing outliers; `none` does not perform any scaling, suitable for pre-normalized data.|
|split_ratio|float (0.0 < x < 1.0)|`0.5`|The ratio of training data against validation data.|
|batch_size|int (x >= 0)|`0`|The size of minibatch. The default uses all records in a single batch.|
|max_epochs|int (x >= 1)|`1000`|The number of epochs to stop training.|
|gpus|int (x >= -1)|`-1`|The number of GPUs to use. By default, use all available GPUs. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices).|
|num_nodes|int (x >= 1)|`1`|The number of compute nodes to use. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes).|
|accelerator|`dp`, `ddp`, `ddp_spawn`, `ddp2`, or `horovod`|`ddp`|Distributed mode. See [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes).|
|loglevel|int (0 <= x <= 2)|`2`|What objects to be logged. `0`: nothing; `1`: metrics; `2`: metrics and models.|


### Run as an MLflow Project

Download the latest [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it.
Then, install MLflow on your conda environment:
```
$ conda install mlflow
```

Run the following command:
```
$ mlflow run https://github.com/rafcc/pytorch-bsf \
  -P data=data.tsv \
  -P label=label.tsv \
  -P degree=3 \
  -P header=0 \
  -P delimiter=' ' \
  -P normalize=none \
  -P split_ratio=0.5 \
  -P batch_size=0 \
  -P max_epochs=1000 \
  -P gpus=-1 \
  -P num_nodes=1 \
  -P accelerator=ddp \
  -P loglevel=2
```
which automatically sets up the environment and runs an experiment:
1. Download the latest pytorch-bsf into a temporary directory.
2. Create a new conda environment and install dependencies in it.
3. Run an experiment on the temporary directory and environment.


## Documents

https://rafcc.github.io/pytorch-bsf/


## Author

RIKEN AIP-FUJITSU Collaboration Center (RAFCC)


## License

MIT
