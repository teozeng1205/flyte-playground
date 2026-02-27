from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from json2vec.core.architecture.custom.packages import Parcel
from json2vec.core.architecture.modules.pool import LearnedQueryCrossAttention
from json2vec.core.architecture.modules.positional import PositionalEncoding
from json2vec.core.structs.tree import Address

if TYPE_CHECKING:
    from json2vec.core.structs.config import Structure


class ContextEncoder(torch.nn.Module):
    def __init__(self, structure: Structure, address: Address):
        super().__init__()

        context = structure.contexts[address]

        self.origin: Address = address
        self.destination: Address = context.parent.address

        layer = torch.nn.TransformerEncoderLayer(
            nhead=context.n_heads,
            d_model=structure.d_model,
            dropout=structure.dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=layer, num_layers=context.n_layers)

        self.pool = LearnedQueryCrossAttention(
            n_context=context.n_outputs,
            d_model=structure.d_model,
            nhead=context.n_heads,
            dropout=structure.dropout,
        )

        self.positional = PositionalEncoding(C=structure.d_model)

    def forward(self, parcels: list[Parcel]) -> Parcel:
        concatenated: torch.Tensor = torch.cat([parcel.payload for parcel in parcels], dim=-2)
        N, *dims, L, C = concatenated.shape
        encoded: torch.Tensor = self.encoder(self.positional(concatenated.reshape(-1, L, C)))
        pooled: torch.Tensor = self.pool(encoded).reshape(N, *dims[:-1], -1, C)

        return Parcel(
            payload=pooled,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )
