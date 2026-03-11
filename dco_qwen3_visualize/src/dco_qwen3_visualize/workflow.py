from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import flyte
import pandas as pd
from flyte.io import File

from dco_qwen3_visualize.config import DCOQwen3VisualizeConfig

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )


image = (
    flyte.Image.from_debian_base((3, 12))
    .with_uv_project("dco_qwen3_visualize/pyproject.toml", uvlock=Path("dco_qwen3_visualize/uv.lock"))
)

worker = flyte.TaskEnvironment(
    name="dco-qwen3-visualize-worker",
    image=image,
    resources=flyte.Resources(cpu=8, memory="40Gi"),
    env_vars={
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    },
)

gpu_worker = flyte.TaskEnvironment(
    name="dco-qwen3-visualize-gpu-worker",
    image=image,
    resources=flyte.Resources(cpu=8, memory="48Gi", gpu="A100:1"),
    env_vars={
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    },
    secrets=[flyte.Secret("HF_TOKEN")],
)

overwatch = flyte.TaskEnvironment(
    name="dco-qwen3-visualize-overwatch",
    image=image,
    env_vars={"PYTHONUNBUFFERED": "1"},
    depends_on=[worker, gpu_worker],
)


@worker.task
def profile_and_sample_dco(
    parquet_uris: list[str],
    customer: str,
    sales_date: str,
    sample_rows: int,
    train_rows: int,
    viz_rows: int,
    output_prefix: str,
    batch_size: int,
) -> tuple[File, File, File, File]:
    from dco_qwen3_visualize.io import sample_parquet_files, write_json
    from dco_qwen3_visualize.sampling import build_representative_samples_from_parquet

    _configure_logging()
    config = DCOQwen3VisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting profile_and_sample_dco for customer=%s sales_date=%s parquet_files=%d sample_rows=%d train_rows=%d viz_rows=%d",
        customer,
        sales_date,
        len(parquet_uris),
        sample_rows,
        train_rows,
        viz_rows,
    )
    started_at = time.perf_counter()
    sample_frame, profile = sample_parquet_files(
        parquet_uris=parquet_uris,
        sample_size=config.sample_rows,
        customer=config.customer,
        sales_date=config.sales_date,
        random_seed=config.random_seed,
        batch_size=config.batch_size,
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="dco_qwen3_sample_"))
    sample_path = tmpdir / "sample.parquet"
    train_path = tmpdir / "context_sample.parquet"
    eval_path = tmpdir / "eval_sample.parquet"
    profile_path = tmpdir / "profile.json"
    sample_frame.to_parquet(sample_path, index=False)
    representative = build_representative_samples_from_parquet(
        sample_path,
        train_rows=min(config.train_rows, len(sample_frame)),
        viz_rows=min(config.viz_rows, len(sample_frame)),
        random_seed=config.random_seed,
        batch_size=config.batch_size,
        config=config,
        train_output_path=train_path,
        viz_output_path=eval_path,
    )
    profile["representative_sampling"] = representative
    write_json(profile_path, profile)
    LOGGER.info(
        "Completed profile_and_sample_dco: sampled_rows=%d train_rows=%d viz_rows=%d elapsed=%.2fs",
        len(sample_frame),
        representative["train_rows"],
        representative["viz_rows"],
        time.perf_counter() - started_at,
    )
    return (
        File.from_local_sync(str(sample_path)),
        File.from_local_sync(str(train_path)),
        File.from_local_sync(str(eval_path)),
        File.from_local_sync(str(profile_path)),
    )


@gpu_worker.task
def fit_qwen3_artifacts(
    context_sample_file: File,
    eval_sample_file: File,
    customer: str,
    sales_date: str,
    sample_rows: int,
    train_rows: int,
    viz_rows: int,
    output_prefix: str,
    batch_size: int,
) -> tuple[File, File, File, File, File, File]:
    from dco_qwen3_visualize.io import write_json
    from dco_qwen3_visualize.model import run_qwen3_visualization

    _configure_logging()
    config = DCOQwen3VisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting fit_qwen3_artifacts for customer=%s sales_date=%s train_rows=%d viz_rows=%d",
        customer,
        sales_date,
        train_rows,
        viz_rows,
    )
    started_at = time.perf_counter()
    context_frame = pd.read_parquet(context_sample_file.download_sync())
    eval_frame = pd.read_parquet(eval_sample_file.download_sync())
    tmpdir = Path(tempfile.mkdtemp(prefix="dco_qwen3_artifacts_"))
    result = run_qwen3_visualization(context_frame, eval_frame, config, tmpdir)

    viz_path = tmpdir / "viz_sample.parquet"
    pretrained_path = tmpdir / "pretrained_embeddings.parquet"
    finetuned_path = tmpdir / "finetuned_embeddings.parquet"
    metrics_path = tmpdir / "metrics.json"
    result["viz_frame"].to_parquet(viz_path, index=False)
    result["pretrained_frame"].to_parquet(pretrained_path, index=False)
    result["finetuned_frame"].to_parquet(finetuned_path, index=False)
    write_json(metrics_path, result["metrics"])
    LOGGER.info(
        "Completed fit_qwen3_artifacts: viz_rows=%d elapsed=%.2fs",
        len(result["viz_frame"]),
        time.perf_counter() - started_at,
    )
    return (
        File.from_local_sync(str(viz_path)),
        File.from_local_sync(str(pretrained_path)),
        File.from_local_sync(str(finetuned_path)),
        File.from_local_sync(str(result["finetune_pairs_path"])),
        File.from_local_sync(str(result["adapter_tar_path"])),
        File.from_local_sync(str(metrics_path)),
    )


@worker.task
def render_artifacts(
    sample_file: File,
    context_sample_file: File,
    eval_sample_file: File,
    profile_file: File,
    viz_sample_file: File,
    pretrained_embeddings_file: File,
    finetuned_embeddings_file: File,
    finetune_pairs_file: File,
    adapter_tar_file: File,
    metrics_file: File,
    customer: str,
    sales_date: str,
    sample_rows: int,
    train_rows: int,
    viz_rows: int,
    output_prefix: str,
    batch_size: int,
    run_timestamp: str,
    upload_urls: dict[str, str],
) -> dict[str, str]:
    import pandas as pd

    from dco_qwen3_visualize.io import artifact_uri, read_json, upload_artifacts_via_presigned_urls, write_json
    from dco_qwen3_visualize.render import render_standalone_dashboard

    _configure_logging()
    config = DCOQwen3VisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting render_artifacts for customer=%s sales_date=%s upload_targets=%d",
        customer,
        sales_date,
        len(upload_urls),
    )
    profile_path = profile_file.download_sync()
    sample_path = sample_file.download_sync()
    context_path = context_sample_file.download_sync()
    eval_path = eval_sample_file.download_sync()
    viz_path = viz_sample_file.download_sync()
    pretrained_path = pretrained_embeddings_file.download_sync()
    finetuned_path = finetuned_embeddings_file.download_sync()
    finetune_pairs_path = finetune_pairs_file.download_sync()
    adapter_path = adapter_tar_file.download_sync()
    metrics_path = metrics_file.download_sync()

    profile = read_json(profile_path)
    metrics = read_json(metrics_path)
    viz_frame = pd.read_parquet(viz_path)

    artifact_dir = Path(tempfile.mkdtemp(prefix="dco_qwen3_render_"))
    html_path = artifact_dir / "dashboard.html"
    manifest_path = artifact_dir / "manifest.json"
    destination_prefix = config.run_output_prefix(run_timestamp)

    html = render_standalone_dashboard(
        frame=viz_frame,
        hover_columns=metrics["hover_columns"],
        customer=config.customer,
        sales_date=config.sales_date,
        profile=profile,
        total_rows=profile["total_rows"],
        parquet_file_count=profile["parquet_file_count"],
        hours_present=profile["hours_present"],
        metrics=metrics,
    )
    html_path.write_text(html, encoding="utf-8")

    uploads = {
        "profile": artifact_uri(destination_prefix, "profile.json"),
        "sample": artifact_uri(destination_prefix, "sample.parquet"),
        "context_sample": artifact_uri(destination_prefix, "context_sample.parquet"),
        "eval_sample": artifact_uri(destination_prefix, "eval_sample.parquet"),
        "viz_sample": artifact_uri(destination_prefix, "viz_sample.parquet"),
        "pretrained_embeddings": artifact_uri(destination_prefix, "pretrained_embeddings.parquet"),
        "finetuned_embeddings": artifact_uri(destination_prefix, "finetuned_embeddings.parquet"),
        "finetune_pairs": artifact_uri(destination_prefix, "finetune_pairs.jsonl"),
        "finetuned_adapter": artifact_uri(destination_prefix, "finetuned_adapter.tar.gz"),
        "metrics": artifact_uri(destination_prefix, "metrics.json"),
        "dashboard": artifact_uri(destination_prefix, "dashboard.html"),
        "manifest": artifact_uri(destination_prefix, "manifest.json"),
    }
    manifest = {
        "customer": config.customer,
        "sales_date": config.sales_date,
        "input_uri": config.input_uri,
        "run_output_prefix": destination_prefix,
        "run_timestamp": run_timestamp,
        "train_rows": metrics["train_rows"],
        "viz_rows": metrics["viz_rows"],
        "pair_count": metrics["pair_count"],
        "feature_columns": metrics["feature_columns"],
        "hover_columns": metrics["hover_columns"],
        "artifacts": uploads,
    }
    write_json(manifest_path, manifest)

    filename_to_local_path = {
        "profile.json": profile_path,
        "sample.parquet": sample_path,
        "context_sample.parquet": context_path,
        "eval_sample.parquet": eval_path,
        "viz_sample.parquet": viz_path,
        "pretrained_embeddings.parquet": pretrained_path,
        "finetuned_embeddings.parquet": finetuned_path,
        "finetune_pairs.jsonl": finetune_pairs_path,
        "finetuned_adapter.tar.gz": adapter_path,
        "metrics.json": metrics_path,
        "dashboard.html": str(html_path),
        "manifest.json": str(manifest_path),
    }
    upload_artifacts_via_presigned_urls(filename_to_local_path, upload_urls)
    return {
        "dashboard_uri": uploads["dashboard"],
        "manifest_uri": uploads["manifest"],
        "viz_sample_uri": uploads["viz_sample"],
        "pretrained_embeddings_uri": uploads["pretrained_embeddings"],
        "finetuned_embeddings_uri": uploads["finetuned_embeddings"],
    }


@overwatch.task
def execute(
    customer: str = "AA",
    sales_date: str = "2026-03-07",
    sample_rows: int = 100_000,
    train_rows: int = 50_000,
    viz_rows: int = 50_000,
    output_prefix: str = "s3://3v-teo-dev/dco_visualize/",
    batch_size: int = 50_000,
    run_timestamp: str = "",
    parquet_uris: list[str] | None = None,
    sample_file: File | None = None,
    context_sample_file: File | None = None,
    eval_sample_file: File | None = None,
    profile_file: File | None = None,
    upload_urls: dict[str, str] | None = None,
) -> dict[str, str]:
    _configure_logging()
    config = DCOQwen3VisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting execute for customer=%s sales_date=%s sample_rows=%d train_rows=%d viz_rows=%d staged_inputs=%s",
        customer,
        sales_date,
        sample_rows,
        train_rows,
        viz_rows,
        bool(sample_file and context_sample_file and eval_sample_file and profile_file),
    )
    if sample_file is None or context_sample_file is None or eval_sample_file is None or profile_file is None:
        if not parquet_uris:
            raise ValueError("parquet_uris must be provided when staged inputs are absent")
        sample_file, context_sample_file, eval_sample_file, profile_file = profile_and_sample_dco(
            parquet_uris=parquet_uris,
            customer=customer,
            sales_date=sales_date,
            sample_rows=sample_rows,
            train_rows=train_rows,
            viz_rows=viz_rows,
            output_prefix=output_prefix,
            batch_size=batch_size,
        )

    viz_sample_file, pretrained_embeddings_file, finetuned_embeddings_file, finetune_pairs_file, adapter_tar_file, metrics_file = fit_qwen3_artifacts(
        context_sample_file=context_sample_file,
        eval_sample_file=eval_sample_file,
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )

    if not upload_urls:
        raise ValueError("upload_urls must be provided for remote artifact publication")
    return render_artifacts(
        sample_file=sample_file,
        context_sample_file=context_sample_file,
        eval_sample_file=eval_sample_file,
        profile_file=profile_file,
        viz_sample_file=viz_sample_file,
        pretrained_embeddings_file=pretrained_embeddings_file,
        finetuned_embeddings_file=finetuned_embeddings_file,
        finetune_pairs_file=finetune_pairs_file,
        adapter_tar_file=adapter_tar_file,
        metrics_file=metrics_file,
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=output_prefix,
        batch_size=batch_size,
        run_timestamp=run_timestamp,
        upload_urls=upload_urls,
    )
