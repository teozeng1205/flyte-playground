from typing import Callable

import pluggy

from json2vec.core.processors.spec import PluginSpec

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="processors")

pm.add_hookspecs(module_or_class=PluginSpec)

PROCESSORS: dict[str, Callable[[dict], dict]] = {}


def register(func):
    name = func.__name__

    if name in PROCESSORS:
        raise ValueError(f"Processor '{name}' is already registered.")

    PROCESSORS[name] = func

    return func
