import torch


def jitter(inputs: torch.Tensor, jitter: torch.Tensor) -> torch.Tensor:
    jitter = torch.rand_like(inputs).sub(torch.rand_like(inputs)).mul(jitter)

    return inputs.add(jitter)


def test_jitter():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    jitter_amount = torch.tensor(0.1)

    jittered = jitter(inputs, jitter_amount)

    assert isinstance(jittered, torch.Tensor)
    assert jittered.shape == inputs.shape
    assert torch.all(jittered <= inputs + jitter_amount)
    assert torch.all(jittered >= inputs - jitter_amount)
    assert not torch.allclose(jittered, inputs), "Jittered tensor should differ from inputs"
