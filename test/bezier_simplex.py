import torch_bsf.bezier_simplex as bs


def test_indices():
  for i in bs.indices(1, 1):
      assert i == (1,)
