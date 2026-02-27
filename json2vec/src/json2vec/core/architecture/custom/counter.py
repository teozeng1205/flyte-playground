import torch

from json2vec.core.structs.tree import Address


class Counter(torch.nn.Module):
    def __init__(self, address: Address, size: int):
        super().__init__()

        self.size: int = size

        # init with ones to avoid division by zero
        # it doesn't matter much since we will normalize over time
        self.register_buffer("counts", torch.ones(size, dtype=torch.int64))
        self.is_full: bool = False

    @torch.no_grad()
    def forward(self, values: torch.Tensor):
        if self.training and not self.is_full:
            could_overflow = self.counts.max().add(values.numel()).gt(torch.iinfo(self.counts.dtype).max)

            if could_overflow:
                # if we are approaching the max value, we stop counting and assume the counts are full
                self.is_full = True
                return values

            self.counts += torch.bincount(values.view(-1), minlength=self.counts.shape[0]).to(self.counts.dtype)

        return values

    @property
    @torch.no_grad()
    def weight(self) -> torch.Tensor:
        return self.counts.sum() / (self.counts * self.counts.numel())


def test_counter():
    counter = Counter(address=Address("test"), size=5)
    data = torch.tensor([0, 1, 2, 2, 3, 4, 4, 4])
    counter(data)
    assert torch.all(counter.counts == torch.tensor([1, 1, 2, 1, 3]).add(1))
    weight = counter.weight
    assert weight.shape[0] == 5

    assert torch.isclose(weight.sum(), torch.tensor(5.4167))
