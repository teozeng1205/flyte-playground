import shutil
import traceback
from collections import defaultdict
from functools import cache, partialmethod, wraps
from typing import Any, NotRequired, Self, TypedDict

import lightning.pytorch as lit
import numpy as np
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict

from json2vec.core.architecture.custom.packages import Embedding, Parcel, Prediction
from json2vec.core.architecture.modules.encoder import ContextEncoder
from json2vec.core.architecture.modules.node import NodeModule
from json2vec.core.data.datasets import dataloader, mock
from json2vec.core.logging.config import info
from json2vec.core.structs.enums import Metric, Stage, Strata, TensorKey
from json2vec.core.structs.experiment import Operations, Session
from json2vec.core.structs.tree import Address
from json2vec.core.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)


class Output(TypedDict):
    loss: NotRequired[torch.Tensor]
    predictions: list[Prediction]


@beartype
def step(
    module: "JSON2Vec",
    batch: TensorDict[Address, TensorFieldBase],
    batch_idx: int,
    strata: Strata,
) -> Output:
    predictions: list[Prediction] = module.forward(batch)

    if strata == Strata.predict:
        return Output(predictions=predictions)

    losses: list[torch.Tensor] = []

    for prediction in predictions:

        if isinstance(prediction, Embedding):
            continue

        address: Address = prediction.address
        request: RequestBase = module.session.structure.requests[address]
        extension: Plugin = TENSORFIELDS[request.type]

        loss: torch.Tensor = extension.loss(module=module, prediction=prediction, batch=batch[address], strata=strata)
        losses.append(loss * torch.tensor(request.weight))

    if len(losses) == 0:
        # under idealistic circumstances this would never happen.
        # but with small mask rates, batch sizes, and flat input data it is possible
        logger.warning("no trainable fields in batch, returning zero loss")
        loss: torch.Tensor = torch.tensor(0.0, device=batch.device, requires_grad=True)
        return Output(loss=loss, predictions=[])

    loss: torch.Tensor = module.track((Metric.loss, strata), value=torch.stack(losses).sum())

    return Output(loss=loss, predictions=predictions)


def compile(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        try:
            import thunder

            model = thunder.compile(model)
            logger.info("successfully compiled module with thunder")
        except Exception:
            traceback.print_exc()
            logger.info("[thunder] Returning uncompiled model instead.")
        return model

    return wrapper


@cache
def groupname(names: tuple[str, ...]) -> str:
    assert len(names) > 1

    group, *keys = tuple(map(lambda x: x.replace("/", ":").lower(), names))

    key: str = ":".join(list(keys))

    return f"{group}/{key}"


class JSON2Vec(lit.LightningModule):
    @beartype
    def __init__(self, session: Session, operations: Operations):

        super().__init__()

        self.session: Session = session
        self.operations: Operations = operations

        self.nodes: torch.nn.ModuleDict[str, NodeModule] = torch.nn.ModuleDict()

        for address in self.session.structure.requests | self.session.structure.contexts:
            self.nodes[address] = NodeModule(structure=self.session.structure, address=address)

        self.example_input_array = mock(structure=session.structure)

    def track(self, names: tuple[str, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.log(
            name=groupname(names),
            value=value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.session.structure.batch_size,
        )

        return value

    @beartype
    def forward(self, inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        for address in self.session.structure.requests.keys():
            if address in self.session.pruned:
                continue

            tensorfield: TensorFieldBase = inputs[address]
            embedder: EmbedderBase = self.nodes[address].embedder
            embedding: Parcel = embedder(tensorfield)
            processed[embedding.destination].append(embedding)
            outgoing[embedding.origin] = embedding

            if address in self.session.output:
                predictions.append(Embedding.from_parcel(embedding))

        # DAG traversal from leaves to root
        for depth in reversed(self.session.structure.depthwise):
            # these are order-independent within the same depth
            for address in depth:
                encoder: ContextEncoder = self.nodes[address].encoder
                encoding: Parcel = encoder(processed[address])
                processed[encoding.destination].append(encoding)
                outgoing[encoding.origin] = encoding

                if address in self.session.output:
                    predictions.append(Embedding.from_parcel(encoding))

        for address in self.session.structure.requests.keys():

            if (torch.any(inputs[address].trainable)) or (address in self.session.pruned):

                heritage: list[Address] = self.session.structure.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address] for address in heritage
                    if address not in self.session.pruned
                ]

                decoder: DecoderBase = self.nodes[address].decoder
                predictions.append(decoder(parcels))

        return predictions

    @beartype
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.session.learning_rate,
        )

    def on_save_checkpoint(self, checkpoint):
        for key, model in [("operations", Operations), ("session", Session)]:
            checkpoint[key] = getattr(self, key).model_dump()

    def on_load_checkpoint(self, checkpoint):
        for key, model in [("operations", Operations), ("session", Session)]:
            if key in checkpoint and getattr(self, key) is None:
                setattr(self, key, model.model_validate(checkpoint[key]))
            if getattr(self, key) is None:
                raise ValueError(f"missing {key} in checkpoint and constructor")

    @classmethod
    def get_or_create(
        cls,
        session: Session|None = None,
        operations: Operations | None = None,
        checkpoint: str | None = None,
    ) -> Self:

        if checkpoint is None:

            model: "JSON2Vec" = cls(session=session, operations=operations)

            return model

        else:
            state = torch.load(checkpoint, weights_only=False, map_location="cpu")

            model: "JSON2Vec" = cls.load_from_checkpoint(
                checkpoint,
                session=session or Session.model_validate(state["session"]),
                operations=operations or Operations.model_validate(state["operations"]),
                strict=False
            )

            return model

    def write(self, predictions: list[Prediction]) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:

        supervised: dict[Address, dict[str, Any]] = {}
        embeddings: dict[Address, dict[str, Any]] = {}

        for prediction in predictions:

            if isinstance(prediction, Embedding):

                embedding: np.ndarray = prediction.payload[TensorKey.embedding].detach().cpu().numpy()
                embeddings[prediction.address] = {TensorKey.embedding.name: embedding}

                continue

            request: RequestBase = self.session.structure.requests[prediction.address]

            extension: Plugin = TENSORFIELDS[request.type]

            scribed: dict[TensorKey, Any]|None = extension.write(module=self, prediction=prediction)

            if scribed is not None:
                supervised[prediction.address] = scribed



        return supervised, embeddings

    training_step = partialmethod(step, strata=Strata.train)
    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)

    train_dataloader = partialmethod(dataloader, strata=Strata.train)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate)
    test_dataloader = partialmethod(dataloader, strata=Strata.test)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict)


    # FIXME these methods should go in a dedicated callback
    on_train_epoch_start = partialmethod(info, hook="start", strata=Strata.train)
    on_train_epoch_end = partialmethod(info, hook="end", strata=Strata.train)
    on_validation_epoch_start = partialmethod(info, hook="start", strata=Strata.validate)
    on_validation_epoch_end = partialmethod(info, hook="end", strata=Strata.validate)
    on_test_epoch_start = partialmethod(info, hook="start", strata=Strata.test)
    on_test_epoch_end = partialmethod(info, hook="end", strata=Strata.test)
    on_prediction_epoch_start = partialmethod(info, hook="start", strata=Strata.predict)
    on_prediction_epoch_end = partialmethod(info, hook="end", strata=Strata.predict)
