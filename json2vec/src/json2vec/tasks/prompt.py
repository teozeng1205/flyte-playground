import logging
import os
from datetime import datetime
from pathlib import Path

import flyte
import torch
from flyte._task import TaskTemplate
from flyte.io import Dir, File
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from rich import print
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from json2vec.core.architecture.custom.write import Writer
from json2vec.core.architecture.modules.root import JSON2Vec
from json2vec.core.logging.throughput import ThroughputLogger
from json2vec.core.structs.enums import Stage
from json2vec.core.structs.experiment import Experiment, Operations, Session
from json2vec.orchestration.environments import overwatch, worker


def build(model: JSON2Vec, callbacks: list[Callback], names: list[str] | None=None) -> Trainer:

    return Trainer(
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else None,
        logger=model.operations.logger(*names) if names is not None else False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        **model.session.trainer
    )


@worker.task
def fit(
    names: list[str],
    session: Session|None=None,
    operations: Operations|None=None,
    checkpoint: File|None=None,
    patch: list[dict[str, str]]|None=None
) -> File:

    inpath: str|None = checkpoint.download_sync() if checkpoint is not None else None
    model: JSON2Vec = JSON2Vec.get_or_create(session=session, operations=operations, checkpoint=inpath)

    if patch is not None:
        model.session.patch(patch)

    filename: str = f"{model.session.structure.name}-{model.session.name}-" + "{epoch}-{step}-{val_loss:.2f}"
    checkpointer: ModelCheckpoint = ModelCheckpoint(dirpath="./models/", filename=filename, monitor="loss/validate")
    callbacks: list[Callback] =[ThroughputLogger(), checkpointer]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.fit(model=model)
    return File.from_local_sync(local_path=str(checkpointer.best_model_path))


@worker.task
def validate(
    names: list[str],
    checkpoint: File,
    session: Session|None=None,
    operations: Operations|None=None,
    patch: list[dict[str, str]]|None=None
):

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, operations=operations, checkpoint=checkpoint.download_sync())

    if patch is not None:
        model.session.patch(patch)

    callbacks: list[Callback] = [ThroughputLogger()]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.validate(model=model)


@worker.task
def test(
    names: list[str],
    checkpoint: File,
    session: Session|None=None,
    operations: Operations|None=None,
    patch: list[dict[str, str]]|None=None
):

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, operations=operations, checkpoint=checkpoint.download_sync())

    if patch is not None:
        model.session.patch(patch)

    callbacks: list[Callback] = [ThroughputLogger()]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.test(model=model)


@worker.task
def predict(
    session: Session|None,
    operations: Operations|None,
    names: list[str]|None,
    checkpoint: File,
    patch: list[dict[str, str]]|None=None
) -> Dir:

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, operations=operations, checkpoint=checkpoint.download_sync())

    if patch is not None:
        model.session.patch(patch)

    os.makedirs(name=(outpath := "tmp/predictions"), exist_ok=True)
    callbacks: list[Callback] = [Writer(outpath)]
    trainer: Trainer = build(model=model, callbacks=callbacks)
    trainer.predict(model=model, return_predictions=False)
    return Dir.from_local_sync(local_path=str(outpath))


@overwatch.task
def execute(experiment: Experiment) -> dict[str, File|Dir]:

    checkpoint: File|None = None
    if experiment.checkpoint is not None:
        checkpoint: File = File.from_existing_remote(experiment.checkpoint)

    tasks: dict[Stage, TaskTemplate] = dict(fit=fit, validate=validate, test=test, predict=predict) # type: ignore
    names: list[str] = [experiment.project, experiment.name, experiment.notes]

    outputs: dict[str, File|Dir] = {}

    for session in experiment.sessions:

        task: TaskTemplate = tasks[session.task].override(short_name=session.name)

        outputs[session.name] = output = task(
            session=session,
            operations=experiment.operations,
            checkpoint=checkpoint,
            names=names
        )

        if isinstance(output, File):
            checkpoint: File = output
    
    return outputs




if __name__ == "__main__":

    cwd: Path = Path.cwd()

    flyte.init_from_config(cwd / ".flyte" / "config.yaml", root_dir=cwd / "src", log_level=logging.ERROR)

    experiment: Experiment = Experiment.from_config("experiments")

    if experiment.remote:

        url: str = flyte.run(execute, experiment=experiment).url

        panel: Panel = Panel(
            renderable=Text(text="Execution", style=f"link {url}"),
            title=f"{experiment.project} / {experiment.name}",
            subtitle=datetime.now().strftime("%H:%M:%S"),
            padding=(1, 16, 1, 16),
            expand=False,
        )

        print(Padding(pad=1, renderable=panel))

    else:
        execute(experiment=experiment)

