import inspect
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Self, Type, TypeAlias, TypedDict, get_type_hints

import jsonpatch
import pydantic
import yaml
from faker import Faker
from lightning.pytorch.loggers import Logger
from rich.prompt import Prompt
from yaml.loader import SafeLoader

from json2vec.core.processors.base import PROCESSORS
from json2vec.core.structs.enums import Stage, Strata, Suffix
from json2vec.core.structs.structure import Structure
from json2vec.core.structs.tree import Address

fake = Faker()



def forge(cls: Type) -> TypeAlias:

    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)

    namespace: dict[str, Any] = {
        k: hints.get(k, v.annotation)
        for k, v in sig.parameters.items()
        if k != "self"
    }
    return TypedDict(f"{cls.__name__}Parameters", namespace, total=False)

# TrainerParameters: TypeAlias = forge(lit.Trainer)
# AdamWParameters: TypeAlias = forge(torch.optim.AdamW)

class Dataset(pydantic.BaseModel):

    root: str
    sample_rate: Annotated[float, pydantic.Field(gt=0.0, le=1.0, default=1.0)]
    file_buffer_size: Annotated[int, pydantic.Field(gt=0)]
    observation_buffer_size: Annotated[int, pydantic.Field(gt=0)]
    processor: Annotated[str, pydantic.Field(default="default")]
    kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    suffix: Suffix
    patterns: dict[Strata, str]

    @pydantic.model_validator(mode="after")
    def check_processor_registered(self):

        if self.processor is not None and self.processor not in PROCESSORS:
            raise ValueError(f"you haven't registered processor {self.processor}")

        return self



class PatchOp(TypedDict, total=False):
    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str
    value: Any
    from_: str  # "from" is a keyword in Python


class Session(pydantic.BaseModel):

    name: str
    dataset: Dataset
    structure: Structure

    task: Stage
    trainer: dict[str, Any] = pydantic.Field(default_factory=dict)

    pruned: list[Address] = pydantic.Field(default_factory=list)
    reset: list[Address] = pydantic.Field(default_factory=list)
    output: list[Address] = pydantic.Field(default_factory=list)

    learning_rate: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    p_prune: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]
    p_mask: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]

    @pydantic.model_validator(mode="after")
    def check_overriden_fields(self):

        for attribute in ["pruned, reset"]:
            for field in getattr(self, attribute, []):
                if field not in self.structure.requests:
                    raise ValueError(f"{attribute} field '{field}' not found in structure requests")

        for attribute in ["output"]:
            for field in getattr(self, attribute, []):
                if field not in self.structure.contexts and field not in self.structure.requests:
                    raise ValueError(f"{attribute} context '{field}' not found in structure contexts or requests")

        return self

    def patch(self, ops: Iterable[PatchOp], *, in_place: bool = True) -> Self:

        ops = list(ops)
        data = self.model_dump(mode="python")
        patched = jsonpatch.apply_patch(data, ops, in_place=in_place)

        return self.__class__.model_validate(patched)



class Loading(pydantic.BaseModel):
    num_workers: Annotated[int | None, pydantic.Field(ge=0, default=None)]
    persistent_workers: Annotated[bool, pydantic.Field(default=True)]
    pin_memory: Annotated[bool, pydantic.Field(default=True)]


# class Debugging(pydantic.BaseModel):
#     trainable_fields_only: Annotated[bool, pydantic.Field(default=True)]


class Operations(pydantic.BaseModel):

    tracking: Literal["neptune", "wandb"] | None = None
    loading: Loading
    # logging: ...
    # debugging: Debugging
    # executing: ...

    def logger(self, project: str, run: str, notes: str) -> Logger|bool:

       match self.tracking:

            case "neptune":
                from lightning.pytorch.loggers import NeptuneLogger
                logger = NeptuneLogger(project=project, name=run)
                run = logger.experiment
                run["sys/notes"] = notes
                return logger

            case "wandb":
                from lightning.pytorch.loggers import WandbLogger
                logger = WandbLogger(project=project, name=run)
                run = logger.experiment
                run.notes = notes
                return logger

            case _:
                return False

class Experiment(pydantic.BaseModel):

    project: str
    remote: bool
    checkpoint: str|None = None

    name: Annotated[str, pydantic.Field(
        default_factory=lambda: f"{fake.word()}-{fake.color_name().lower()}"
    )]

    notes: Annotated[str, pydantic.Field(default="")]

    sessions: list[Session]
    operations: Operations

    def prompt(self):

        self.name = Prompt.ask(
            "[bold green]Run Name[/bold green]",
            default=self.name,
            show_default=True,
        )

        self.notes: str = Prompt.ask(
            "[bold green]Run Notes[/bold green]",
            default=self.notes,
            show_default=False,
        )
    
    @classmethod
    def from_config(cls, pathname: str, name: str|None=None) -> Self:

        if name is None:
            experiments: list[str] = [file.stem for file in Path(pathname).glob("*.yaml")]

            name = Prompt.ask(
                "[bold green]Experiment Name[/bold green]",
                choices=experiments,
                default=experiments[1],
            )

        cfg  = Loader.from_experiment(pathname, name)
        experiment: Self = cls.model_validate(cfg)
        experiment.prompt()

        return experiment


class IncludeRef:
    def __init__(self, namespace: str, name: str):
        self.namespace = namespace
        self.name = name

    def __repr__(self):
        return f"IncludeRef({self.namespace}:{self.name})"


def include_constructor(loader, node):
    raw = loader.construct_scalar(node).strip()
    namespace, name = raw.split("/", 1)
    return IncludeRef(namespace, name)

SafeLoader.add_constructor("!include", include_constructor)


class Loader:
    def __init__(self, base_dir: str | Path = "configuration"):
        self.base = Path(base_dir)

    def load_yaml(self, path: Path):
        with path.open() as f:
            return yaml.load(f, Loader=SafeLoader)

    def resolve(self, obj, seen=None):
        if seen is None:
            seen = set()

        # --- Handle IncludeRef ---
        if isinstance(obj, IncludeRef):
            file_path = self.base / obj.namespace / f"{obj.name}.yaml"

            if not file_path.exists():
                raise FileNotFoundError(f"Include not found: {file_path}")

            # if file_path in seen:
            #     raise ValueError(f"Circular include detected: {file_path}")

            seen.add(file_path)
            included = self.load_yaml(file_path)
            return self.resolve(included, seen)

        # --- Handle dict ---
        if isinstance(obj, dict):
            return {k: self.resolve(v, seen) for k, v in obj.items()}

        # --- Handle list ---
        if isinstance(obj, list):
            return [self.resolve(v, seen) for v in obj]

        # --- Other types ---
        return obj

    def load(self, name: str):
        raw = self.load_yaml(self.base / f"{name}.yaml")
        return self.resolve(raw)
    
    @classmethod
    def from_experiment(cls, path: str, name: str):
        return cls(path).load(name)

