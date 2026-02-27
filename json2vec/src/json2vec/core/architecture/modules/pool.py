import torch


# TODO could simplify this later but whatever
class LearnedTensor(torch.nn.Module):
    def __init__(self, *sizes: int):
        super().__init__()

        for dim in sizes:
            assert isinstance(dim, int), "each dim must be of type int"
            assert dim >= 1, "each dim must be greater than 0"

        self.values = torch.nn.Parameter(torch.normal(mean=0.0, std=1e-2, size=sizes))

    def forward(self, N: int) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(N, *self.values.shape)


def test_learned_tensor():
    sizes = (3, 4, 5)
    learned_tensor = LearnedTensor(*sizes)

    assert isinstance(learned_tensor.values, torch.nn.Parameter)
    assert learned_tensor.values.shape == sizes

    output = learned_tensor(1)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, *sizes)
    assert output.dtype == torch.float32
    assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-2)
    assert torch.allclose(output.std(), torch.tensor(1e-2), atol=1e-2)

    # Test that the values are learnable
    optimizer = torch.optim.Adam(learned_tensor.parameters(), lr=0.01)
    initial_values = learned_tensor.values.clone()

    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    assert not torch.equal(learned_tensor.values, initial_values), (
        "LearnedTensor values should change after optimization"
    )


class LearnedQueryCrossAttention(torch.nn.Module):
    def __init__(self, n_context: int, d_model: int, nhead: int, dropout: float):
        super().__init__()

        self.queries = LearnedTensor(n_context, d_model)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        N, L, C = memory.shape

        queries = self.queries(N)

        out, _ = self.attention(queries, memory, memory, need_weights=False)

        return self.norm(out)


def test_pool():
    n_context = 5
    d_model = 16
    nhead = 4
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    model = LearnedQueryCrossAttention(n_context, d_model, nhead, dropout)

    memory = torch.randn(batch_size, seq_length, d_model)

    output = model(memory)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, n_context, d_model)
