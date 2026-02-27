from datetime import timedelta
from pathlib import Path

import flyte

# FIXME make this configurable

version = "0.2.2"
image = flyte.Image.from_debian_base().with_uv_project("pyproject.toml", uvlock=Path("uv.lock"))
reusable = flyte.ReusePolicy(replicas=(1, 5), idle_ttl=timedelta(minutes=10))

worker: flyte.TaskEnvironment = flyte.TaskEnvironment(
    name="json2vec",
    image=image,
    reusable=reusable,
    # resources=flyte.Resources(cpu=32, memory="240Gi", shm="4Gi", gpu="A100:1"),
    resources=flyte.Resources(cpu=16, memory="200Gi", shm="4Gi", gpu="T4:1"),
    secrets=[flyte.Secret("WANDB_API_KEY")],
    env_vars=({"PYTHONUNBUFFERED": "1"}),
    cache=flyte.Cache(behavior="auto", ignored_inputs=("operations", "names"), serialize=True, version_override=version),
)

overwatch: flyte.TaskEnvironment = flyte.TaskEnvironment(
    name="overwatch",
    image=image,
    # reusable=reusable,
    env_vars=({"PYTHONUNBUFFERED": "1"}),
    depends_on=[worker],
)
