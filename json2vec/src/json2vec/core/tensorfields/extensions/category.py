from __future__ import annotations

from multiprocessing import Manager
from multiprocessing.managers import ListProxy, SyncManager
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterable, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from ordered_set import OrderedSet
from tensordict import TensorDict, tensorclass

from json2vec.core.architecture.custom.counter import Counter
from json2vec.core.architecture.custom.packages import Parcel, Prediction
from json2vec.core.data.processing import pad_nested
from json2vec.core.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.core.structs.tree import Address
from json2vec.core.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)

if TYPE_CHECKING:
    from json2vec.core.architecture.modules.root import JSON2Vec
    from json2vec.core.structs.experiment import Session, Structure

category: Plugin = Plugin(name="category")


def apply(values: Any, func: Callable, *args, **kwargs) -> Any:
    if isinstance(values, Iterable) and not isinstance(values, str):
        return [apply(v, func, *args, **kwargs) for v in values]

    elif values is None:
        return None

    else:
        return func(values, *args, **kwargs)

class Vocabulary:
    
    def __init__(self, master: ListProxy, lock: Lock):

        self.master: ListProxy[str] = master
        self.lock: Lock = lock
        self.vocab: OrderedSet[str] = OrderedSet(list(master))

    def __call__(self, word: str) -> list[int]:

        # if the word is already known, then just tokenize it
        if word in self.vocab:
            return self.vocab.index(word)

        # OK, it is not known locally... We will lock the global state and update the local vocab
        with self.lock:
            self.vocab: OrderedSet[str] = OrderedSet(list(self.master))

            if word not in self.vocab:
                self.vocab.add(word)
                self.master.append(word)

        # now, it should _definitely_ be known.
        return self.vocab.index(word)

class OnlineVocabularyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.manager: SyncManager = Manager()
        self.master: ListProxy[str] = self.manager.list()
        self.lock: Lock = self.manager.Lock()

    def _save_to_state_dict(self, state_dict, prefix, keep_vars):
        super()._save_to_state_dict(state_dict, prefix, keep_vars)

        state_dict[prefix + "vocabulary"] = list(self.master)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs
    ):

        vocab: list[str] = state_dict.pop(prefix + "vocabulary")
        self.master: ListProxy[str] = self.manager.list(vocab)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs
        )
    
    @property
    def state(self) -> Vocabulary:

        return Vocabulary(master=self.master, lock=self.lock)


@category.register
class Request(RequestBase):
    name: str
    type: Literal["category"]
    query: str
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)]
    max_vocab_size: Annotated[int, pydantic.Field(gt=0, default=10_000)]
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)]
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)]
    top_k: Annotated[list[int], pydantic.Field(default_factory=list)]

    @pydantic.model_validator(mode="after")
    def check_top_k(self):
        for top_k in self.top_k:
            if not isinstance(top_k, int):
                raise ValueError("top_k values must be integers")

            if top_k <= 0:
                raise ValueError("top_k values must be positive")

            if top_k == 1:
                raise ValueError("top_k values must not be 1")

            if top_k >= self.max_vocab_size:
                raise ValueError("top_k values must be less than max_vocab_size")

        return self

    

@category.register
@tensorclass
class TensorField(TensorFieldBase):
    content: torch.Tensor
    trainable: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        values: list,
        address: Address,
        session: Session,
        strata: Strata,
        state: Vocabulary,
    ) -> TensorFieldBase:

        shape: tuple[int, ...] = session.structure.shapes[address]

        tokens = apply(values, state)

        data, states = pad_nested(
            nested=tokens,
            shape=tuple([len(values), *shape[1:-1]]),
            dtype=np.int64,
            pad_value=0,
        )

        if data.max() > (max_vocab_size := session.structure.requests[address].max_vocab_size):
            print(
                f"Token in address {address} exceeds max vocab size of {max_vocab_size}"
            )

        states = torch.tensor(states, dtype=torch.int64)

        content = torch.tensor(data=data).add(len(Tokens)).masked_scatter(states != Tokens.valued.value, states)


        return cls(
            content=content,
            trainable=torch.zeros_like(input=content, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def mask(self, p_mask: float):
        mask_token = torch.full_like(input=self.content, fill_value=Tokens.masked.value)
        is_masked = torch.rand_like(input=self.content, dtype=torch.float).lt(other=p_mask)

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.content = self.content.masked_scatter(is_masked, mask_token)

        self.trainable |= is_masked

    def prune(self, p_prune: float = 1.0):
        prune_tokens = torch.full_like(input=self.content, fill_value=Tokens.pruned)

        is_pruned = (
            torch.rand(self.content.size(0), *([1] * (len(self.content.shape) - 1)), device=self.content.device)
            .lt(p_prune)
            .expand_as(self.content)
        )

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.content = self.content.masked_scatter(is_pruned, prune_tokens)

        self.trainable |= is_pruned

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        structure: Structure,
    ):
        shape: tuple[int, ...] = (batch_size, *structure.shapes[address][1:1])

        content = torch.full(shape, Tokens.pruned)

        return cls(
            content=content,
            trainable=torch.zeros_like(input=content, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@category.register
class Embedder(EmbedderBase):
    def __init__(self, structure: Structure, address: Address):
        super().__init__(structure=structure, address=address)

        request: Request = structure.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.max_vocab_size: int = request.max_vocab_size + len(Tokens)

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel()

        self.embeddings = torch.nn.Embedding(
            num_embeddings=request.max_vocab_size + len(Tokens),
            embedding_dim=structure.d_model,
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: tuple[int, ...]

        N, *dims = inputs.content.shape

        if (inputs.content >= self.max_vocab_size).any().item():
            print(
                f"Token in address {self.origin} exceeds max vocab size of {self.max_vocab_size}"
            )

        reshaped: torch.Tensor = inputs.content.reshape(-1)

        embeddings: torch.Tensor = self.embeddings(reshaped).reshape(N, *dims, -1)


        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def state(self) -> Vocabulary:
        return self.vocab.state



@category.register
class Decoder(DecoderBase):
    def __init__(self, structure: Structure, address: Address):
        super().__init__(structure=structure, address=address)

        self.address: Address = address
        request: RequestBase = structure.requests[address]

        vocab_size: int = request.max_vocab_size + len(Tokens)

        self.linear = torch.nn.Linear(
            in_features=structure.d_model,
            out_features=vocab_size,
        )

        self.counter = Counter(address=address, size=vocab_size)

    @beartype
    def forward(self, parcels: list[Parcel]) -> Prediction:
        # FIXME much of this can be shared with done in the decoder base
        N, *_, C = parcels[0].payload.shape

        stacked: torch.Tensor = torch.cat([parcel.payload.reshape(N, -1, C) for parcel in parcels], dim=1)

        pooled: torch.Tensor = self.pool(self.positional(stacked))

        payload: TensorDict[TensorKey, torch.Tensor] = TensorDict(
            source={
                TensorKey.content: self.linear(pooled),
            }
        )

        return Prediction(
            payload=payload,
            address=self.address,
            batch_size=pooled.shape[0],
        )


@category.register
def loss(
    module: JSON2Vec,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    N: int = batch.targets[TensorKey.content].numel()
    trainable = batch.trainable.reshape(N)


    loss: torch.Tensor = (
        torch.nn.functional.cross_entropy(
            input=(inputs := prediction.payload[TensorKey.content].reshape(N, -1)),
            target=(targets := batch.targets[TensorKey.content].reshape(N)),
            weight=module.nodes[prediction.address].decoder.counter.weight,
            reduction="none",
        )
        .mul(trainable)
        .mean()
    )

    for top_k in module.session.structure.requests[prediction.address].top_k:
        module.track(
            (prediction.address, strata, Metric.accuracy, f"top_{top_k}"),
            value=1 - (
                inputs
                .topk(k=top_k, dim=1)
                .indices.eq(targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(trainable).float().mean()    
            )
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=1 - inputs.argmax(dim=1).eq(targets).masked_select(trainable).float().mean(),
    )

    return loss


@category.register
def write(module: JSON2Vec, prediction: Prediction):

    node = module.nodes[prediction.address]

    vocab = np.concatenate([
        np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens)),
        np.array(node.embedder.vocab.state),
    ])

    narrow: torch.Tensor = prediction.payload[TensorKey.content].narrow(dim=-1, start=0, length=len(vocab))

    requested_k: int = max(node.request.top_k, default=1)
    top_k: int = min(requested_k, narrow.shape[-1])

    probabilities: torch.Tensor = narrow.softmax(dim=-1)
    top_k_values, top_k_indices = probabilities.topk(k=top_k, dim=-1)

    top_k_indices_np: np.ndarray = top_k_indices.detach().cpu().numpy().astype(np.int32)
    top_k_labels_np: np.ndarray = vocab[top_k_indices_np]
    top_k_probabilities_np: np.ndarray = top_k_values.detach().float().cpu().numpy()

    top_k_dicts = np.empty(top_k_labels_np.shape[:-1], dtype=object)
    flat_dicts = top_k_dicts.reshape(-1)
    flat_labels = top_k_labels_np.reshape(-1, top_k_labels_np.shape[-1])
    flat_probs = top_k_probabilities_np.reshape(-1, top_k_probabilities_np.shape[-1])

    for i in range(flat_dicts.shape[0]):
        flat_dicts[i] = {str(label): float(prob) for label, prob in zip(flat_labels[i], flat_probs[i])}

    return {
        TensorKey.content.name: vocab[narrow.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)],
        TensorKey.top_k.name: top_k_dicts,
        TensorKey.probability.name: narrow.max(dim=-1).values.detach().float().cpu().numpy(),
    }
