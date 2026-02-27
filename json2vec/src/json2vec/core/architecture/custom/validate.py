import torch

from json2vec.core.tensorfields.base import TensorFieldBase


def validate(inputs: TensorFieldBase) -> None:
    if inputs.content.shape != inputs.state.shape:
        raise ValueError("values and indicators must always have the same shape")

    if not torch.all(inputs.content.mul(inputs.state).eq(0.0), dim=None):
        raise ValueError("values should be imputed if not null, padded, or masked")

    if not torch.all(inputs.content.le(1.0), dim=None):
        raise ValueError("values should be less than or equal to 1.0")

    if not torch.all(inputs.content.ge(0.0), dim=None):
        raise ValueError("values should be greater than or equal to 0.0")
