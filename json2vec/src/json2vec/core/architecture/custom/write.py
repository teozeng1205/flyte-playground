from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from lightning.pytorch import callbacks
from tensordict import TensorDict

from json2vec.core.architecture.custom.packages import Prediction
from json2vec.core.structs.enums import TensorKey
from json2vec.core.structs.tree import Address
from json2vec.core.tensorfields.base import TensorFieldBase

if TYPE_CHECKING:
    from json2vec.core.architecture.modules.root import JSON2Vec



class Writer(callbacks.BasePredictionWriter):

    def __init__(self, path: os.PathLike | str):

        super().__init__(write_interval="batch")

        self.path: os.PathLike = path
        self.schema: pa.schema|None = None
        self.writer: pq.ParquetWriter|None = None

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: JSON2Vec,
        output: dict[str, list[Prediction]],
        batch_indices: list[int]|None,
        batch: TensorDict[Address, TensorFieldBase],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        supervised: dict[Address, dict[TensorKey, Any]]
        embeddings: dict[Address, dict[TensorKey, Any]]

        supervised, embeddings = pl_module.write(predictions=output["predictions"])

        cols: list[pl.DataFrame] = []

        for address, values in supervised.items():
            cols.append(
                pl.DataFrame(data={tensorkey: pl.Series(name=value) for tensorkey, value in values.items()})
                .select(pl.struct([pl.col(name=tensorkey) for tensorkey in values.keys()]).alias(name=address))
            )

        supervised: pl.DataFrame = pl.concat(items=cols, how="horizontal")
        items = [
            pl.from_records(data=batch["metadata"], schema=["inputs"], orient="row"),
            supervised.select(pl.struct(supervised.columns).alias(name="predictions")),
        ]

        if len(embeddings) > 0:
            cols: list[pl.DataFrame] = []
            for address, values in embeddings.items():
                cols.append(
                    pl.DataFrame(data={tensorkey: pl.Series(name=value) for tensorkey, value in values.items()})
                    .select(pl.struct([pl.col(name=tensorkey) for tensorkey in values.keys()]).alias(name=address))
                )

            embeddings: pl.DataFrame = pl.concat(items=cols, how="horizontal")
            items.append(embeddings.select(pl.struct(embeddings.columns).alias(name="embeddings")))


        table: pa.table = pl.concat(
            items=items,
            how="horizontal"
        ).to_arrow()

        if self.writer is None:

            self.schema: pa.schema = table.schema

            self.writer: pq.ParquetWriter = pq.ParquetWriter(
                where=os.path.join(self.path, f"rank-{trainer.local_rank}.parquet"),
                schema=self.schema
            )

        self.writer.write_table(table.cast(self.schema))

        if hasattr(self.writer, 'flush'):
            self.writer.flush()

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if self.writer:
            self.writer.close()
            self.writer: None = None
