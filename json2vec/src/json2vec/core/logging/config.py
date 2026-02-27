import json
import sys
from typing import Literal

from loguru import logger
from rich.console import Console
from rich.json import JSON

from json2vec.core.structs.enums import Strata

console = Console(file=sys.stdout)


def sink(message):
    record = message.record
    extras = {k: str(v) for k, v in record["extra"].items()}
    payload = {
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "level": record["level"].name,
        **extras,
        "message": record["message"],
    }
    # Pretty-print JSON with color
    # console.print(JSON(json.dumps(payload), indent=None))


logger.remove()
logger.add(sink)


def info(self, strata: Strata, hook: Literal["start", "end"]):
    logger.bind(
        source="lightning",
        rank=self.global_rank,
        epoch=self.current_epoch,
        step=self.global_step,
        hook=hook,
        strata=str(strata),
    ).info(f"{hook}ing {strata} epoch")
