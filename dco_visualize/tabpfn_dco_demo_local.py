from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.io import collect_parquet_metadata, list_s3_parquet_objects, sample_parquet_files, write_json
from dco_visualize.model import fit_embedding_model, transform_parquet_file, write_embedding_bundle
from dco_visualize.render import render_standalone_dashboard, save_dashboard_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DCO TabPFN 2.5 demo locally.")
    parser.add_argument("--customer", default="AA")
    parser.add_argument("--sales-date", default="2026-03-07")
    parser.add_argument("--sample-rows", type=int, default=5_000)
    parser.add_argument("--train-rows", type=int, default=2_500)
    parser.add_argument("--viz-rows", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "demo_outputs"))
    parser.add_argument("--source-aws-profile", default="3VPROD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_aws_profile:
        os.environ["AWS_PROFILE"] = args.source_aws_profile

    config = DCOVisualizeConfig(
        customer=args.customer,
        sales_date=args.sales_date,
        sample_rows=args.sample_rows,
        train_rows=min(args.train_rows, args.sample_rows, 50_000),
        viz_rows=min(args.viz_rows, args.sample_rows),
        batch_size=args.batch_size,
    )
    config.validate()

    print(f"Enumerating {config.input_uri}")
    parquet_uris = list_s3_parquet_objects(config.input_uri)
    if not parquet_uris:
        raise ValueError(f"No parquet files found under {config.input_uri}")
    parquet_objects = collect_parquet_metadata(parquet_uris)
    total_rows = sum(item.row_count for item in parquet_objects)
    print(f"Found {len(parquet_uris)} parquet files and {total_rows:,} rows")

    sample_frame, profile = sample_parquet_files(
        parquet_uris=parquet_uris,
        sample_size=min(config.sample_rows, total_rows),
        customer=config.customer,
        sales_date=config.sales_date,
        random_seed=config.random_seed,
        batch_size=config.batch_size,
        parquet_objects=parquet_objects,
    )

    output_dir = Path(args.output_dir) / f"{config.customer}_{config.sales_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "sample.parquet"
    embeddings_path = output_dir / "embeddings_full.parquet"
    viz_path = output_dir / "viz_sample.parquet"
    aggregate_path = output_dir / "market_aggregates.parquet"
    bundle_path = output_dir / "embedding_bundle.pt"
    metrics_path = output_dir / "metrics.json"
    profile_path = output_dir / "profile.json"
    dashboard_path = output_dir / "dashboard.html"

    sample_frame.to_parquet(sample_path, index=False)
    write_json(profile_path, profile)

    train_frame = sample_frame.sample(n=min(config.train_rows, len(sample_frame)), random_state=config.random_seed)
    fit_result = fit_embedding_model(train_frame.reset_index(drop=True), config)
    viz_frame, aggregate_frame, metrics = transform_parquet_file(
        model=fit_result.model,
        parquet_path=sample_path,
        output_path=embeddings_path,
        viz_rows=config.viz_rows,
        batch_size=config.batch_size,
        random_seed=config.random_seed,
    )
    viz_frame.to_parquet(viz_path, index=False)
    aggregate_frame.to_parquet(aggregate_path, index=False)
    write_embedding_bundle(bundle_path, fit_result.model)

    merged_metrics = dict(fit_result.metrics)
    merged_metrics.update(metrics)
    merged_metrics["train_rows"] = int(len(train_frame))
    write_json(metrics_path, merged_metrics)

    image_paths = save_dashboard_images(
        frame=viz_frame,
        aggregate_frame=aggregate_frame,
        output_dir=output_dir,
        customer=config.customer,
        sales_date=config.sales_date,
    )
    html = render_standalone_dashboard(
        frame=viz_frame,
        aggregate_frame=aggregate_frame,
        hover_columns=merged_metrics["hover_columns"],
        customer=config.customer,
        sales_date=config.sales_date,
        total_points=len(viz_frame),
        total_rows=profile["total_rows"],
        parquet_file_count=profile["parquet_file_count"],
        hours_present=profile["hours_present"],
        metrics=merged_metrics,
        image_paths=image_paths,
    )
    dashboard_path.write_text(html, encoding="utf-8")

    print(f"Demo output: {output_dir}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Bundle: {bundle_path}")


if __name__ == "__main__":
    main()
