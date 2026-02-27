import datetime
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import flyte
import flyte.remote
import json2vec.tasks.metrics as _metrics
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from flyte.io import Dir, File

image = (
    flyte.Image.from_debian_base()
    .with_uv_project("pyproject.toml", uvlock=Path("uv.lock"))
    .with_source_folder(Path(__file__).parent / "metrics", dst="metrics_scripts", copy_contents_only=True)
)

overwatch: flyte.TaskEnvironment = flyte.TaskEnvironment(
    name="refitter",
    image=image,
    env_vars={"PYTHONUNBUFFERED": "1"},
    resources=flyte.Resources(cpu=8, memory="32Gi"),
)

_FOLDER = "estream_hive"
_MARKETS = sorted([
    "US-GB", "US-PR", "US-MX", "US-KR",
    "US-JP", "US-IT", "US-ES", "US-DO",
    "US-CR", "US-CO", "US-BR", "PR-US",
    "NO-NO", "MX-US", "JP-US", "GB-US",
    "DO-US", "CR-US", "CO-US", "BR-US",
], reverse=True)


@overwatch.task
async def fetch_3v_data(date: str) -> Dir:
    """Pull a 5 % holdout evaluation slice for *date* using the shuffle pipeline."""
    day = date.replace("-", "_")
    eval_path = (
        f"s3://atp-sbmlops-use1-pipeline-data-store/shopping_data_filtered/"
        f"many_markets/{_FOLDER}_eval_shuffled/{day}"
    )
    shuffle = flyte.remote.Task.get(
        project="flytesnacks",
        domain="development",
        name="dask-data-shuffler.start_dask_task",
        version="bcc28887e17804f83146c25e5d87cf00",
    )
    await shuffle(
        input_base_path="s3://s3-atp-digitaldata-shoppingdata-use1-shopping-data-processed/shopping_data_response/",
        intermidate_output_path=(
            f"s3://atp-sbmlops-use1-pipeline-data-store/shopping_data_filtered/"
            f"many_markets/{_FOLDER}_eval_gathered/{day}"
        ),
        combined_parquet_output_path=(
            f"s3://atp-sbmlops-use1-pipeline-data-store/shopping_data_filtered/"
            f"many_markets/{_FOLDER}_eval_combined/{day}"
        ),
        markets=_MARKETS,
        days=[date],
        shuffled_output_path=eval_path,
        frac=0.05,
    )
    return Dir.from_existing_remote(eval_path)


@overwatch.task
async def validate_model_predictions(predictions: Dir) -> bool:
    """Run all four evaluate scripts and return True if overall thresholds pass."""
    metrics_dir = Path("/root/metrics_scripts")

    files: list[File] = await predictions.list_files()
    parquet_files = [f for f in files if f.name.endswith(".parquet")]
    if not parquet_files:
        print("No parquet files found.")
        return False

    tmpdir = Path(tempfile.mkdtemp())

    # Sample up to 500k rows total across shards to stay within memory limits
    max_rows = 500_000
    rows_per_file = max(1_000, max_rows // len(parquet_files))
    tables = []
    for f in parquet_files:
        t = pq.read_table(await f.download())
        if len(t) > rows_per_file:
            step = len(t) // rows_per_file
            t = t.take(list(range(0, len(t), step))[:rows_per_file])
        tables.append(t)
    combined = str(tmpdir / "predictions.parquet")
    pq.write_table(pa.concat_tables(tables), combined)
    print(f"Sampled {sum(len(t) for t in tables):,} rows from {len(parquet_files)} files")

    cityloc = str(metrics_dir / "citylocation.csv")
    scripts = [
        ("calculate_metrics.py",            []),
        ("calculate_metrics_cabin.py",       []),
        ("calculate_metrics_country_od.py",  [cityloc]),
        ("calculate_metrics_all.py",         [cityloc]),
    ]
    for script, extra in scripts:
        out = str(tmpdir / script.replace(".py", ".csv"))
        result = subprocess.run(
            [sys.executable, str(metrics_dir / script), combined, *extra, out],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"{script} failed (exit {result.returncode}):\n{result.stderr}")
        df = pd.read_csv(out)
        print(f"\n=== {script.replace('.py', '').replace('_', ' ')} ===")
        print(df.to_string(index=False))

    base = pd.read_csv(str(tmpdir / "calculate_metrics.csv"))
    overall_abs_acc10 = base["abs_acc10"].mean()
    overall_acc_pct10 = base["acc_pct10"].mean()
    print(f"\nOverall abs_acc10={overall_abs_acc10:.4f}  acc_pct10={overall_acc_pct10:.4f}")

    passed = bool(overall_abs_acc10 >= 0.80 and overall_acc_pct10 >= 0.70)
    print(f"Validation {'PASSED' if passed else 'FAILED'}")
    return passed


@overwatch.task(triggers=flyte.Trigger("refit taxml every tuesday", flyte.Cron("0 0 * * 3")))
async def refit():
    day = (datetime.date.today() - datetime.timedelta(days=2)).strftime("%Y-%m-%d")

    kwargs = {
        "input_base_path": "s3://s3-atp-digitaldata-shoppingdata-use1-shopping-data-processed/shopping_data_response/",
        "intermidate_output_path": f"s3://atp-sbmlops-use1-pipeline-data-store/shopping_data_filtered/many_markets/{_FOLDER}_gathered/{day.replace('-', '_')}",
        "combined_parquet_output_path": f"s3://atp-sbmlops-use1-pipeline-data-store/shopping_data_filtered/many_markets/{_FOLDER}_combined/{day.replace('-', '_')}",
        "markets": _MARKETS,
        "days": [day],
        "shuffled_output_path": f"s3://atp-sbmlops-use1-training-data/shopping_data_filtered/many_markets/{_FOLDER}_shuffled/{day.replace('-', '_')}",
        "frac": 0.1,
    }

    shuffle = flyte.remote.Task.get(
        project="flytesnacks",
        domain="development",
        name="dask-data-shuffler.start_dask_task",
        version="bcc28887e17804f83146c25e5d87cf00",
    )
    await shuffle(**kwargs)

    shuffled = Dir.from_existing_remote(str(kwargs["shuffled_output_path"]))
    patch = [{"op": "replace", "path": "/dataset/root", "value": f"{shuffled.path}"}]

    checkpoints: list[File] = await (
        Dir.from_existing_remote("s3://union-persistent-6fe3d20a0ed633f3/taxml-model-history/")
        .list_files()
    )
    checkpoint = max(
        [c for c in checkpoints if c.name.endswith(".ckpt")],
        key=lambda c: c.name,
    )

    fit = flyte.remote.Task.get(
        project="flytesnacks",
        domain="development",
        name="worker.fit",
        version="307c2fbe2d27b93df2f0ad229feaefd7",
    )

    print(f"Using checkpoint: {checkpoint}")
    refitted = await fit(
        checkpoint=checkpoint,
        session=None,
        operations=None,
        patch=patch,
        names=["taxml", day, f"refit on {day}"],
    )

    print(f"Saving new checkpoint for {day}")
    downloaded = await refitted.download()
    new_checkpoint = await File.from_local(
        downloaded,
        remote_destination=f"s3://union-persistent-6fe3d20a0ed633f3/taxml-model-history/{day}.ckpt",
    )

    # Fetch holdout eval slice, run predict, validate
    eval_data = await fetch_3v_data(date=day)
    eval_patch = [{"op": "replace", "path": "/dataset/root", "value": eval_data.path}]

    predict = flyte.remote.Task.get(
        project="flytesnacks",
        domain="development",
        name="worker.predict",
        version="307c2fbe2d27b93df2f0ad229feaefd7",  # TODO: pin correct predict version
    )
    predictions = await predict(
        session=None,
        operations=None,
        names=["taxml", day, f"validate on {day}"],
        checkpoint=new_checkpoint,
        patch=eval_patch,
    )

    passed = await validate_model_predictions(predictions=predictions)
    if not passed:
        print(f"WARNING: model validation failed for {day}")


if __name__ == "__main__":
    cwd = Path.cwd()
    flyte.init_from_config(cwd / ".flyte" / "config.yaml", root_dir=cwd / "src", log_level=logging.ERROR)
    run = flyte.run(
        validate_model_predictions,
        predictions=Dir.from_existing_remote(
            "s3://union-us-east-1-atpco/b8/atpco/flytesnacks/development/r56b5swt848vrqq8z8xw/8peyurznkthc4uk37lh7077yu/1/nw/r56b5swt848vrqq8z8xw-8peyurznkthc4uk37lh7077yu-0/956b1d7fa62428923b58dd643bbe71ea/predictions"
        ),
    )
    print(run.url)
