
import torch
from tensordict import TensorClass, TensorDict

from json2vec.core.structs.enums import TensorKey
from json2vec.core.structs.tree import Address


class Parcel(TensorClass):
    payload: torch.Tensor
    origin: Address
    destination: Address | None


def test_parcel():
    parcel = Parcel(
        payload=torch.randn(2, 3, 4),
        origin="input",
        destination="output",
        batch_size=[2],
    )

    assert isinstance(parcel.payload, torch.Tensor)
    assert parcel.payload.shape == (2, 3, 4)
    assert parcel.origin == "input"
    assert parcel.destination == "output"


# @jaxtyped(typechecker=beartype)
class Prediction(TensorClass):
    address: Address
    payload: TensorDict[TensorKey, torch.Tensor]

class Embedding(Prediction):

    @classmethod
    def from_parcel(cls, parcel: Parcel) -> "Prediction":
        return cls(
            address=parcel.origin,
            payload=TensorDict(
                {TensorKey.embedding: parcel.payload},
                batch_size=parcel.payload.shape[0],
            )
        )


def test_prediction():
    prediction = Prediction(
        address="output",
        payload=TensorDict(
            {
                TensorKey.content: torch.randn(2, 3),
                TensorKey.state: torch.randint(0, 2, (2, 3), dtype=torch.int8),
            },
            batch_size=[2],
        ),
    )

    assert prediction.address == "output"
    assert isinstance(prediction.payload, TensorDict)
    assert isinstance(prediction.payload[TensorKey.content], torch.Tensor)
    assert prediction.payload[TensorKey.content].shape == (2, 3)
    assert isinstance(prediction.payload[TensorKey.state], torch.Tensor)
    assert prediction.payload[TensorKey.state].shape == (2, 3)

