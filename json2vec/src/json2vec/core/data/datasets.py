from __future__ import annotations

import hashlib
import os
import random
import re
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from functools import cache, partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import jmespath
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from beartype import beartype
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from json2vec.core.data.processing import Pipeline
from json2vec.core.processors.base import PROCESSORS
from json2vec.core.structs.enums import Strata, Suffix
from json2vec.core.structs.experiment import Session, Structure

# import pyarrow.fs as fs
from json2vec.core.structs.tree import Address
from json2vec.core.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.core.architecture.modules.root import JSON2Vec


@beartype
def sha256(string: str, bits: int = 64) -> int:
    if not (1 <= bits <= 256):
        raise ValueError("bits must be between 1 and 256")

    # Hash the string using SHA-256
    h: bytes = hashlib.sha256(string.encode("utf-8")).digest()

    # Convert hash to integer and truncate to desired number of bits
    return int.from_bytes(h, "big") >> (256 - bits)


def test_sha256():
    assert sha256("test", 32) == 2676412545
    assert sha256("test", 64) == 11495104353665842533
    assert sha256("test", 128) == 212047248112658246449511647784264716309


@beartype
def fetch(session: Session, strata: Strata) -> Iterator[str]:

    regex: re.Pattern = re.compile(session.dataset.patterns[strata])

    parsed = urlparse(session.dataset.root)

    if parsed.scheme == "s3":
        fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
        path = f"{parsed.netloc}{parsed.path}"
        uri_prefix = "s3://"

    elif parsed.scheme in ("", "file"):
        fs = pafs.LocalFileSystem()
        path = parsed.path
        uri_prefix = ""

    else:
        raise ValueError(f"Unsupported scheme: {parsed.scheme or 'file'}")

    selector = pafs.FileSelector(path, recursive=True)

    worker_info = get_worker_info()

    if worker_info is None:
        worker_id: int = 0
        num_workers: int = 1
    else:
        worker_id: int = worker_info.id
        num_workers: int = worker_info.num_workers

    for info in fs.get_file_info(selector):
        if info.is_file:
            uri_path: str = f"{uri_prefix}{info.path}" if uri_prefix else info.path
            if regex is None or regex.search(uri_path):
                # if sha256(uri_path) % num_workers == worker_id:
                yield uri_path


@beartype
def read(pipe: Iterable[str], session: Session) -> Iterator[dict]:
    match session.dataset.suffix:
        case Suffix.ndjson:
            import json

            for path in pipe:
                with open(path, "r") as file:
                    for line in file:
                        if line.strip():
                            yield json.loads(line)

        case Suffix.feather | Suffix.parquet | Suffix.avro | Suffix.csv | Suffix.orc | Suffix.json:
            for path in pipe:
                parsed = urlparse(path)

                if parsed.scheme == "s3":
                    fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
                    path = f"{parsed.netloc}{parsed.path}"

                elif parsed.scheme in ("", "file"):
                    fs = pafs.LocalFileSystem()
                    path = parsed.path

                else:
                    raise ValueError(f"Unsupported scheme: {parsed.scheme or 'file'}")

                bucket = parsed.netloc
                key = parsed.path.lstrip("/")  # remove leading slash

                # Create pyarrow S3 filesystem

                try:
                    dataset = ds.dataset(
                        f"{bucket}/{key}",
                        format=session.dataset.suffix.value,
                        filesystem=fs,
                    )

                    for batch in dataset.to_batches(batch_size=1024):
                        yield from batch.to_pylist()
                except Exception:
                    print(f"Error reading {path}, skipping.")
                    continue

        case _:
            raise ValueError(f"Unsupported suffix: {dataset.suffix}")


@beartype
def process(pipe: Iterable[dict], session: Session) -> Iterator[dict]:
    if session.dataset.processor is None:
        yield from pipe

    else:
        processor = PROCESSORS[session.dataset.processor]

        for item in pipe:
            processed = processor(item, **session.dataset.kwargs)
            if processed is not None:
                yield processed


@beartype
def batch(pipe: Iterable[dict], session: Session) -> Iterator[list[dict]]:
    # FIXME I can just use itertools.batch

    batch: list[dict] = []

    for item in pipe:
        batch.append(item)
        if len(batch) == session.structure.batch_size:
            yield batch
            batch.clear()

    if batch:
        yield batch



@beartype
def shuffle(pipe: Iterable[dict], size: int, strata: Strata) -> Iterator[dict]:

    if strata == Strata.predict:
        yield from pipe
        return

    buffer = deque()

    iterable = iter(pipe)

    for _ in range(size):
        try:
            buffer.append(next(iterable))
        except StopIteration:
            break

    while buffer:
        idx = random.randrange(len(buffer))
        val = buffer[idx]

        try:
            buffer[idx] = next(iterable)
        except StopIteration:
            buffer.remove(val)

        yield val


@beartype
@cache
def query(expression: str) -> jmespath.parser.ParsedResult:
    return jmespath.compile(expression=expression)


def test_query():
    expr = query("foo.bar")
    result = expr.search({"foo": {"bar": 42}})
    assert result == 42


_jmespath_counter = Counter()


def spotcheck(result, address: Address, every: int = 1000):
    _jmespath_counter[address] += 1
    count = _jmespath_counter[address]

    if count % every != 0:
        return  # skip check

    # Fast non-recursive emptiness check
    stack = [result]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif item not in (None, "", [], {}):
            return

    raise ValueError(f"JMESPath query returned empty result for address: {address}")


def encode(
    batch: dict[str, Any],
    session: Session,
    strata: Strata,
    state: dict[Address, Any],
) -> TensorDict[Address, TensorFieldBase]:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in session.structure.requests.items():
        TensorField: type[TensorFieldBase] = TENSORFIELDS[request.type].TensorField

        if (strata == Strata.predict) & (address in session.pruned):
            # basically, if we are in inference mode, we should create empty values

            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                structure=session.structure,
            )

            continue

        result: list = query(request.query).search(batch)

        spotcheck(result=result, address=address)

        out[address] = TensorField.new(
            values=result,
            address=address,
            session=session,
            strata=strata,
            state=state.get(address),
        )

        # otherwise, we should "prune" them entirely during model training
        # but we still need to instantiate them for backpropagation
        if address in session.pruned:
            out[address].prune(p_prune=1.0)

    inputs: TensorDict[Address, TensorFieldBase] = TensorDict(source=out)

    # FIXME this should not happen during real time inference
    if strata == Strata.predict:
        inputs["metadata"] = batch

    
    return inputs


@beartype
def transform(
    pipe: Iterable[dict[str, Any]],
    session: Session,
    strata: Strata,
    state: dict[Address, Any],
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    for batch in pipe:

        yield encode(batch=batch, session=session, strata=strata, state=state)


@beartype
def mask(
    pipe: Iterable[TensorDict[Address, TensorFieldBase]],
    session: Session,
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    if not session.p_mask > 0.0:
        yield from pipe

    else:
        for batch in pipe:
            for address in session.structure.requests.keys():
                field: TensorFieldBase = batch[address]
                field.mask(p_mask=session.p_mask)

            yield batch


@beartype
def prune(
    pipe: Iterable[TensorDict[Address, TensorFieldBase]],
    session: Session,
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    for batch in pipe:
        for address in session.structure.requests.keys():
            field: TensorFieldBase = batch[address]
            field.prune(p_prune=session.p_prune)

        yield batch


def identity(data: Any) -> Any:
    return data

class BatchDataset(IterableDataset):
    def __init__(self, model: JSON2Vec, strata: Strata):
        super().__init__()

        self.session: Session = model.session

        self.state: dict[Address, Any] = {
            address: node.embedder.state
            for address, node in model.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "state")
        }

        self.strata: Strata = strata

    def __iter__(self):
        yield from (
            Pipeline(session=self.session, strata=self.strata, state=self.state)
            | fetch
            | partial(shuffle, size=self.session.dataset.file_buffer_size)
            | read
            | process
            | partial(shuffle, size=self.session.dataset.observation_buffer_size)
            | batch
            | transform
            | mask
            | prune
        )


def dataloader(module: JSON2Vec, strata: Strata) -> DataLoader:

    return DataLoader(
        dataset=BatchDataset(model=module, strata=strata),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=module.operations.loading.num_workers or os.cpu_count() or 0,
        persistent_workers=module.operations.loading.persistent_workers,
        pin_memory=(
            module.operations.loading.pin_memory
            & (strata != Strata.predict)
            & torch.cuda.is_available()
        ),
    )


def mock(structure: Structure) -> TensorDict[Address, TensorFieldBase]:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in structure.requests.items():

        TensorField: TensorFieldBase = TENSORFIELDS[request.type].TensorField

        out[address] = TensorField.empty(
            batch_size=structure.batch_size, 
            address=address, 
            structure=structure
        )

    return TensorDict(source=out, batch_size=structure.batch_size)


