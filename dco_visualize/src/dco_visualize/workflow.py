from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import flyte
from flyte.io import File

from dco_visualize.config import DCOVisualizeConfig

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )

image = (
    flyte.Image.from_debian_base((3, 12))
    .with_uv_project("dco_visualize/pyproject.toml", uvlock=Path("dco_visualize/uv.lock"))
)

worker = flyte.TaskEnvironment(
    name="dco-visualize-worker",
    image=image,
    resources=flyte.Resources(cpu=8, memory="40Gi"),
    env_vars={
        "PYTHONUNBUFFERED": "1",
        "TABPFN_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TABPFN_MODEL_CACHE_DIR": "/tmp/tabpfn_model_cache",
    },
)

gpu_worker = flyte.TaskEnvironment(
    name="dco-visualize-gpu-worker",
    image=image,
    resources=flyte.Resources(cpu=8, memory="48Gi", gpu="A100:1"),
    env_vars={
        "PYTHONUNBUFFERED": "1",
        "TABPFN_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TABPFN_MODEL_CACHE_DIR": "/tmp/tabpfn_model_cache",
    },
    secrets=[flyte.Secret("HF_TOKEN")],
)

overwatch = flyte.TaskEnvironment(
    name="dco-visualize-overwatch",
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
    embedding_dims: int,
    output_prefix: str,
    batch_size: int,
) -> tuple[File, File, File, File]:
    import pandas as pd

    from dco_visualize.io import sample_parquet_files, write_json
    from dco_visualize.sampling import build_representative_samples_from_parquet

    _configure_logging()
    config = DCOVisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        embedding_dims=embedding_dims,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting profile_and_sample_dco for customer=%s sales_date=%s parquet_files=%d sample_rows=%d train_rows=%d",
        config.customer,
        config.sales_date,
        len(parquet_uris),
        config.sample_rows,
        config.train_rows,
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

    tmpdir = Path(tempfile.mkdtemp(prefix="dco_sample_"))
    sample_path = tmpdir / "sample.parquet"
    train_sample_path = tmpdir / "train_sample.parquet"
    viz_input_path = tmpdir / "viz_input.parquet"
    profile_path = tmpdir / "profile.json"

    sample_frame.to_parquet(sample_path, index=False)
    representative_stats = build_representative_samples_from_parquet(
        sample_path,
        train_rows=min(config.train_rows, len(sample_frame)),
        viz_rows=min(config.viz_rows, len(sample_frame)),
        random_seed=config.random_seed,
        batch_size=config.batch_size,
        config=config,
        train_output_path=train_sample_path,
        viz_output_path=viz_input_path,
    )
    profile["representative_sampling"] = representative_stats
    write_json(profile_path, profile)
    LOGGER.info(
        "Completed sampling: sampled_rows=%d train_rows=%d viz_input_rows=%d parquet_files=%d elapsed=%.2fs",
        len(sample_frame),
        representative_stats["train_rows"],
        representative_stats["viz_rows"],
        profile["parquet_file_count"],
        time.perf_counter() - started_at,
    )
    return (
        File.from_local_sync(str(sample_path)),
        File.from_local_sync(str(train_sample_path)),
        File.from_local_sync(str(viz_input_path)),
        File.from_local_sync(str(profile_path)),
    )


@gpu_worker.task
def fit_embedding_artifacts(
    sample_file: File,
    train_sample_file: File,
    viz_input_file: File,
    customer: str,
    sales_date: str,
    sample_rows: int,
    train_rows: int,
    viz_rows: int,
    embedding_dims: int,
    output_prefix: str,
    batch_size: int,
) -> tuple[File, File, File, File, File]:
    import pandas as pd

    from dco_visualize.io import write_json
    from dco_visualize.model import (
        aggregate_parquet_file,
        fit_embedding_model,
        transform_parquet_file,
        write_embedding_bundle,
    )

    _configure_logging()
    config = DCOVisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        embedding_dims=embedding_dims,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    LOGGER.info(
        "Starting fit_embedding_artifacts for customer=%s sales_date=%s sample_rows=%d train_rows=%d viz_rows=%d",
        config.customer,
        config.sales_date,
        config.sample_rows,
        config.train_rows,
        config.viz_rows,
    )
    started_at = time.perf_counter()

    sample_path = sample_file.download_sync()
    train_sample_path = train_sample_file.download_sync()
    viz_input_path = viz_input_file.download_sync()
    train_frame = pd.read_parquet(train_sample_path)
    LOGGER.info(
        "Downloaded staged inputs: sample_path=%s train_sample_path=%s viz_input_path=%s train_rows=%d",
        sample_path,
        train_sample_path,
        viz_input_path,
        len(train_frame),
    )
    fit_result = fit_embedding_model(train_frame, config)
    LOGGER.info(
        "Finished model fitting: pretrained_dim=%s finetuned_dim=%s",
        fit_result.metrics.get("pretrained_embedding_dim", fit_result.model.pretrained.embedding_dim),
        fit_result.metrics.get("finetuned_embedding_dim", fit_result.model.finetuned.embedding_dim),
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="dco_embeddings_"))
    embeddings_path = tmpdir / "embeddings_full.parquet"
    viz_sample_path = tmpdir / "viz_sample.parquet"
    aggregate_path = tmpdir / "market_aggregates.parquet"
    bundle_path = tmpdir / "embedding_bundle.pt"
    metrics_path = tmpdir / "metrics.json"

    viz_frame, aggregate_frame, transform_metrics = transform_parquet_file(
        model=fit_result.model,
        parquet_path=viz_input_path,
        output_path=embeddings_path,
        viz_rows=min(config.dashboard_point_cap, config.viz_rows),
        batch_size=config.batch_size,
        random_seed=config.random_seed,
    )
    full_day_aggregate_frame = aggregate_parquet_file(
        model=fit_result.model,
        parquet_path=sample_path,
        batch_size=config.batch_size,
    )
    viz_only_views = aggregate_frame[
        aggregate_frame["view"].isin(["segment_size", "segment_fingerprint", "segment_agreement"])
    ].copy()
    aggregate_frame = pd.concat([full_day_aggregate_frame, viz_only_views], ignore_index=True)
    viz_frame.to_parquet(viz_sample_path, index=False)
    aggregate_frame.to_parquet(aggregate_path, index=False)
    write_embedding_bundle(bundle_path, fit_result.model)

    metrics = dict(fit_result.metrics)
    metrics.update(transform_metrics)
    metrics["train_rows"] = int(len(train_frame))
    write_json(metrics_path, metrics)
    LOGGER.info(
        "Completed embedding artifacts: embedded_rows=%d viz_rows=%d aggregate_rows=%d elapsed=%.2fs",
        metrics["embedded_rows"],
        metrics["viz_rows"],
        len(aggregate_frame),
        time.perf_counter() - started_at,
    )

    return (
        File.from_local_sync(str(embeddings_path)),
        File.from_local_sync(str(viz_sample_path)),
        File.from_local_sync(str(aggregate_path)),
        File.from_local_sync(str(bundle_path)),
        File.from_local_sync(str(metrics_path)),
    )


@worker.task
def render_artifacts(
    sample_file: File,
    profile_file: File,
    embeddings_file: File,
    viz_sample_file: File,
    aggregate_file: File,
    bundle_file: File,
    metrics_file: File,
    customer: str,
    sales_date: str,
    sample_rows: int,
    train_rows: int,
    viz_rows: int,
    embedding_dims: int,
    output_prefix: str,
    batch_size: int,
    run_timestamp: str,
    upload_urls: dict[str, str],
) -> dict[str, str]:
    import pandas as pd

    from dco_visualize.io import artifact_uri, read_json, upload_artifacts_via_presigned_urls, write_json
    from dco_visualize.render import render_standalone_dashboard, save_dashboard_images

    _configure_logging()
    config = DCOVisualizeConfig(
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        embedding_dims=embedding_dims,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    config.validate()
    if not upload_urls:
        raise ValueError("upload_urls must be provided for remote artifact publication.")
    LOGGER.info(
        "Starting render_artifacts for customer=%s sales_date=%s upload_targets=%d",
        config.customer,
        config.sales_date,
        len(upload_urls),
    )
    started_at = time.perf_counter()

    profile_path = profile_file.download_sync()
    sample_path = sample_file.download_sync()
    embeddings_path = embeddings_file.download_sync()
    viz_sample_path = viz_sample_file.download_sync()
    aggregate_path = aggregate_file.download_sync()
    bundle_path = bundle_file.download_sync()
    metrics_path = metrics_file.download_sync()

    profile = read_json(profile_path)
    metrics = read_json(metrics_path)
    viz_frame = pd.read_parquet(viz_sample_path)
    aggregate_frame = pd.read_parquet(aggregate_path)
    LOGGER.info(
        "Loaded render inputs: viz_rows=%d aggregate_rows=%d hover_columns=%d",
        len(viz_frame),
        len(aggregate_frame),
        len(metrics.get("hover_columns", [])),
    )

    artifact_dir = Path(tempfile.mkdtemp(prefix="dco_artifacts_"))
    destination_prefix = config.run_output_prefix(run_timestamp)

    image_paths = save_dashboard_images(
        frame=viz_frame,
        aggregate_frame=aggregate_frame,
        output_dir=artifact_dir,
        customer=config.customer,
        sales_date=config.sales_date,
    )

    html = render_standalone_dashboard(
        frame=viz_frame,
        aggregate_frame=aggregate_frame,
        hover_columns=metrics["hover_columns"],
        customer=config.customer,
        sales_date=config.sales_date,
        profile=profile,
        total_points=metrics["viz_rows"],
        total_rows=profile["total_rows"],
        parquet_file_count=profile["parquet_file_count"],
        hours_present=profile["hours_present"],
        metrics=metrics,
        image_paths=image_paths,
    )

    html_path = artifact_dir / "dashboard.html"
    manifest_path = artifact_dir / "manifest.json"
    html_path.write_text(html, encoding="utf-8")

    uploads = {
        "profile": artifact_uri(destination_prefix, "profile.json"),
        "sample": artifact_uri(destination_prefix, "sample.parquet"),
        "embeddings": artifact_uri(destination_prefix, "embeddings_full.parquet"),
        "viz_sample": artifact_uri(destination_prefix, "viz_sample.parquet"),
        "market_aggregates": artifact_uri(destination_prefix, "market_aggregates.parquet"),
        "model": artifact_uri(destination_prefix, "embedding_bundle.pt"),
        "metrics": artifact_uri(destination_prefix, "metrics.json"),
        "dashboard": artifact_uri(destination_prefix, "dashboard.html"),
        "manifest": artifact_uri(destination_prefix, "manifest.json"),
    }
    for local_path in image_paths.values():
        uploads[Path(local_path).name] = artifact_uri(destination_prefix, Path(local_path).name)

    manifest = {
        "customer": config.customer,
        "sales_date": config.sales_date,
        "input_uri": config.input_uri,
        "run_output_prefix": destination_prefix,
        "run_timestamp": run_timestamp,
        "train_rows": metrics["train_rows"],
        "embedded_rows": metrics["embedded_rows"],
        "viz_rows": metrics["viz_rows"],
        "retained_columns": metrics["retained_columns"],
        "excluded_columns": metrics["excluded_columns"],
        "hover_columns": metrics["hover_columns"],
        "model_hyperparameters": {
            "encoder_backend": metrics["encoder_backend"],
            "target_column": metrics["target_column"],
            "pretrained": metrics["pretrained"],
            "finetuned": metrics["finetuned"],
            "pretrained_projection": metrics["pretrained_projection"],
            "finetuned_projection": metrics["finetuned_projection"],
        },
        "image_artifacts": {name: uri for name, uri in uploads.items() if name.endswith(".png")},
        "profile": profile,
        "artifacts": uploads,
    }
    write_json(manifest_path, manifest)
    LOGGER.info(
        "Generated dashboard artifacts in %s and manifest for %s",
        artifact_dir,
        destination_prefix,
    )

    filename_to_local_path = {
        "profile.json": profile_path,
        "sample.parquet": sample_path,
        "embeddings_full.parquet": embeddings_path,
        "viz_sample.parquet": viz_sample_path,
        "market_aggregates.parquet": aggregate_path,
        "embedding_bundle.pt": bundle_path,
        "metrics.json": metrics_path,
        "dashboard.html": str(html_path),
        "manifest.json": str(manifest_path),
    }
    for local_path in image_paths.values():
        filename_to_local_path[Path(local_path).name] = local_path

    LOGGER.info("Uploading %d artifacts via presigned URLs", len(filename_to_local_path))
    upload_artifacts_via_presigned_urls(filename_to_local_path, upload_urls)
    LOGGER.info("Completed render_artifacts in %.2fs", time.perf_counter() - started_at)

    return {
        "manifest_uri": uploads["manifest"],
        "dashboard_uri": uploads["dashboard"],
        "embeddings_uri": uploads["embeddings"],
        "model_uri": uploads["model"],
        "run_output_prefix": destination_prefix,
    }


@overwatch.task
def execute(
    customer: str = "AA",
    sales_date: str = "2026-03-07",
    sample_rows: int = 100_000,
    train_rows: int = 50_000,
    viz_rows: int = 500_000,
    embedding_dims: int = 128,
    output_prefix: str = "s3://3v-teo-dev/dco_visualize/",
    batch_size: int = 50_000,
    run_timestamp: str = "",
    sample_file: File | None = None,
    train_sample_file: File | None = None,
    viz_input_file: File | None = None,
    profile_file: File | None = None,
    parquet_uris: list[str] | None = None,
    upload_urls: dict[str, str] | None = None,
) -> dict[str, str]:
    _configure_logging()
    LOGGER.info(
        "Starting execute for customer=%s sales_date=%s sample_rows=%d train_rows=%d viz_rows=%d staged_inputs=%s",
        customer,
        sales_date,
        sample_rows,
        train_rows,
        viz_rows,
        sample_file is not None and train_sample_file is not None and viz_input_file is not None and profile_file is not None,
    )
    if sample_file is None or train_sample_file is None or viz_input_file is None or profile_file is None:
        if not parquet_uris:
            raise ValueError("Either staged sample/profile inputs or parquet_uris must be provided.")
        LOGGER.info("No staged inputs supplied; sampling from parquet URIs inside Flyte")
        sample_file, train_sample_file, viz_input_file, profile_file = profile_and_sample_dco(
            parquet_uris=parquet_uris,
            customer=customer,
            sales_date=sales_date,
            sample_rows=sample_rows,
            train_rows=train_rows,
            viz_rows=viz_rows,
            embedding_dims=embedding_dims,
            output_prefix=output_prefix,
            batch_size=batch_size,
        )
    else:
        LOGGER.info("Using staged sample/profile inputs provided by submit path")

    LOGGER.info("Submitting fit_embedding_artifacts child task")
    embeddings_file, viz_sample_file, aggregate_file, bundle_file, metrics_file = fit_embedding_artifacts(
        sample_file=sample_file,
        train_sample_file=train_sample_file,
        viz_input_file=viz_input_file,
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        embedding_dims=embedding_dims,
        output_prefix=output_prefix,
        batch_size=batch_size,
    )
    LOGGER.info("Submitting render_artifacts child task")
    return render_artifacts(
        sample_file=sample_file,
        profile_file=profile_file,
        embeddings_file=embeddings_file,
        viz_sample_file=viz_sample_file,
        aggregate_file=aggregate_file,
        bundle_file=bundle_file,
        metrics_file=metrics_file,
        customer=customer,
        sales_date=sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        embedding_dims=embedding_dims,
        output_prefix=output_prefix,
        batch_size=batch_size,
        run_timestamp=run_timestamp,
        upload_urls=upload_urls or {},
    )
