import os
from typing import Any, Callable, TypeAlias

import litserve as ls
import torch
from beartype import beartype
from tensordict import TensorDict

from json2vec.core.architecture.custom.packages import Prediction
from json2vec.core.architecture.modules.root import JSON2Vec
from json2vec.core.data.datasets import encode
from json2vec.core.processors.base import PROCESSORS
from json2vec.core.structs.enums import Strata
from json2vec.core.structs.tree import Address
from json2vec.core.tensorfields.base import TensorFieldBase

Input: TypeAlias = TensorDict[Address, TensorFieldBase]


class Deployment(ls.LitAPI):

    def setup(self, device: torch.device) -> None:
        self.model: JSON2Vec = JSON2Vec.get_or_create(checkpoint=os.environ["CHECKPOINT"])

    @beartype
    def decode_request(self, request: dict) -> Input:

        processor: Callable[[dict], dict] = PROCESSORS[self.model.session.dataset.processor]

        return encode(
            batch=processor(request),
            session=self.model.session,
            strata=Strata.predict,
        )

    @beartype
    def batch(self, inputs: list[Input]) -> Input:
        return torch.stack(inputs, dim=0)

    @torch.no_grad
    @beartype
    def predict(self, data: Input) -> list[Prediction]:
        return self.model.forward(data)

    @beartype
    def encode_response(self, response: list[Prediction]) -> dict[str, Any]:
        return self.model.write(predictions=response)


if __name__ == "__main__":

    server: ls.LitServer = ls.LitServer(
        lit_api=Deployment(
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", 1)),
            batch_timeout=float(os.getenv("BATCH_TIMEOUT", 0.0)),
        ),
        accelerator="cpu",
        track_requests=True,
        workers_per_device=int(os.getenv("N_WORKERS", os.cpu_count())),
    )

    server.run(generate_client_file=False)
