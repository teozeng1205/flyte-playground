from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import flyte
from flyte.config import Config
from flyte.io import File
from flyte.models import ActionPhase

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dco_visualize.config import DCOVisualizeConfig, make_run_timestamp
from dco_visualize.io import (
    collect_parquet_metadata,
    generate_presigned_upload_urls,
    list_s3_parquet_objects,
    materialize_parquet_files,
    sample_parquet_files,
    write_json,
)
from dco_visualize.sampling import build_representative_samples_from_parquet
from dco_visualize.workflow import execute

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the DCO visualization Flyte workflow.")
    parser.add_argument("--customer", default="AA", help="Customer code to visualize.")
    parser.add_argument(
        "--sales-date",
        default=date.today().isoformat(),
        help="Sales date partition in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100_000,
        help="Rows to embed from the partition. Use --full-day to embed the entire day partition.",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=50_000,
        help="Maximum rows used to fit the TabPFN branches before full-partition inference.",
    )
    parser.add_argument(
        "--viz-rows",
        type=int,
        default=500_000,
        help="Representative rows retained for embedding comparison before dashboard downsampling.",
    )
    parser.add_argument(
        "--full-day",
        action="store_true",
        help="Embed every row available under the requested day partition.",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=0,
        help="Unused override retained for compatibility. TabPFN determines embedding width.",
    )
    parser.add_argument(
        "--output-prefix",
        default="s3://3v-teo-dev/dco_visualize/",
        help="Destination prefix for published artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Parquet batch size used during streaming sample extraction and full-day transforms.",
    )
    parser.add_argument(
        "--auth-type",
        default="Pkce",
        choices=["Pkce", "DeviceFlow"],
        help="Flyte auth mode to use for submission.",
    )
    parser.add_argument(
        "--source-aws-profile",
        default="3VPROD",
        help="Local AWS profile used to enumerate source parquet objects before submission.",
    )
    parser.add_argument(
        "--output-aws-profile",
        default="3VDEV",
        help="Local AWS profile used to publish completed artifacts to the destination bucket.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    config = DCOVisualizeConfig(
        customer=args.customer,
        sales_date=args.sales_date,
        sample_rows=args.sample_rows,
        train_rows=args.train_rows,
        viz_rows=args.viz_rows,
        embedding_dims=args.embedding_dims,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
    )
    config.validate()
    if args.source_aws_profile:
        os.environ["AWS_PROFILE"] = args.source_aws_profile

    LOGGER.info("Enumerating source parquet objects from %s", config.input_uri)
    parquet_uris = list_s3_parquet_objects(config.input_uri)
    if not parquet_uris:
        raise ValueError(f"No parquet files found under {config.input_uri}")
    parquet_objects = collect_parquet_metadata(parquet_uris)
    total_rows = sum(item.row_count for item in parquet_objects)
    sample_rows = total_rows if args.full_day else min(config.sample_rows, total_rows)
    train_rows = min(config.train_rows, sample_rows, 50_000)
    config = DCOVisualizeConfig(
        customer=args.customer,
        sales_date=args.sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=min(args.viz_rows, sample_rows),
        embedding_dims=args.embedding_dims,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
    )
    config.validate()
    LOGGER.info("Found %d parquet files and %s rows", len(parquet_uris), f"{total_rows:,}")
    if args.full_day:
        LOGGER.info("Full-day mode enabled: embedding all %s rows", f"{config.sample_rows:,}")
    LOGGER.info("Train rows: %s", f"{config.train_rows:,}")
    LOGGER.info("Viz rows: %s", f"{config.viz_rows:,}")

    LOGGER.info("Building staged input artifacts locally")
    staging_dir = Path(tempfile.mkdtemp(prefix="dco_visualize_submit_"))
    sample_path = staging_dir / "sample.parquet"
    train_sample_path = staging_dir / "train_sample.parquet"
    viz_input_path = staging_dir / "viz_input.parquet"
    profile_path = staging_dir / "profile.json"

    if config.sample_rows >= total_rows:
        materialize_parquet_files(
            parquet_uris=parquet_uris,
            customer=config.customer,
            sales_date=config.sales_date,
            batch_size=config.batch_size,
            output_path=sample_path,
            parquet_objects=parquet_objects,
        )
        profile = {
            "customer": config.customer,
            "sales_date": config.sales_date,
            "source_prefix": parquet_uris[0].rsplit("/", 2)[0] + "/",
            "parquet_file_count": len(parquet_objects),
            "total_rows": total_rows,
            "sample_rows": int(total_rows),
            "train_rows": int(config.train_rows),
            "fit_rows": int(config.train_rows),
            "column_count": 0,
            "hours_present": sorted({item.hour for item in parquet_objects}),
            "row_counts_by_file": [{"uri": item.uri, "rows": item.row_count, "hour": item.hour} for item in parquet_objects],
        }
        representative_stats = build_representative_samples_from_parquet(
            sample_path,
            train_rows=config.train_rows,
            viz_rows=config.viz_rows,
            random_seed=config.random_seed,
            batch_size=config.batch_size,
            config=config,
            train_output_path=train_sample_path,
            viz_output_path=viz_input_path,
        )
        profile["representative_sampling"] = representative_stats
        write_json(profile_path, profile)
        LOGGER.info(
            "Prepared full-day parquet with %s rows and representative train/viz samples train_rows=%s viz_rows=%s",
            f"{config.sample_rows:,}",
            f"{representative_stats['train_rows']:,}",
            f"{representative_stats['viz_rows']:,}",
        )
    else:
        sample_frame, profile = sample_parquet_files(
            parquet_uris=parquet_uris,
            sample_size=config.sample_rows,
            customer=config.customer,
            sales_date=config.sales_date,
            random_seed=config.random_seed,
            batch_size=config.batch_size,
            parquet_objects=parquet_objects,
        )
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
            "Prepared %s staged rows with representative train/viz samples train_rows=%s viz_rows=%s",
            f"{len(sample_frame):,}",
            f"{representative_stats['train_rows']:,}",
            f"{representative_stats['viz_rows']:,}",
        )

    config_path = REPO_ROOT / ".flyte" / "config.yaml"
    flyte_config = Config.auto(config_path)

    LOGGER.info("Initializing Flyte connection")
    LOGGER.info("Flyte auth type: %s", args.auth_type)
    flyte.init(
        org=flyte_config.task.org,
        project=flyte_config.task.project,
        domain=flyte_config.task.domain,
        root_dir=SRC_ROOT,
        endpoint=flyte_config.platform.endpoint,
        insecure=flyte_config.platform.insecure,
        insecure_skip_verify=flyte_config.platform.insecure_skip_verify,
        ca_cert_file_path=flyte_config.platform.ca_cert_file_path,
        auth_type=args.auth_type,
        command=flyte_config.platform.command,
        proxy_command=flyte_config.platform.proxy_command,
        client_id=flyte_config.platform.client_id,
        client_credentials_secret=flyte_config.platform.client_credentials_secret,
        rpc_retries=flyte_config.platform.rpc_retries,
        http_proxy_url=flyte_config.platform.http_proxy_url,
        image_builder=flyte_config.image.builder or "local",
        images=flyte_config.image.image_refs or None,
        source_config_path=config_path,
    )

    run_timestamp = make_run_timestamp()
    destination_prefix = config.run_output_prefix(run_timestamp)
    LOGGER.info("Submitting DCO visualization for customer=%s sales_date=%s", config.customer, config.sales_date)
    LOGGER.info("Run timestamp: %s", run_timestamp)

    upload_urls = generate_presigned_upload_urls(
        destination_prefix=destination_prefix,
        filenames=[
            "profile.json",
            "sample.parquet",
            "embeddings_full.parquet",
            "viz_sample.parquet",
            "market_aggregates.parquet",
            "embedding_bundle.pt",
            "metrics.json",
            "dashboard.html",
            "manifest.json",
            "pretrained_embedding_density.png",
            "finetuned_embedding_density.png",
            "route_network.png",
            "fare_calendar.png",
            "market_matrix.png",
            "segment_fingerprint.png",
        ],
        profile_name=args.output_aws_profile,
    )
    LOGGER.info("Generated %d presigned upload URLs", len(upload_urls))
    run = flyte.with_runcontext(copy_style="all").run(
        execute,
        customer=config.customer,
        sales_date=config.sales_date,
        sample_rows=config.sample_rows,
        train_rows=config.train_rows,
        viz_rows=config.viz_rows,
        embedding_dims=config.embedding_dims,
        output_prefix=config.output_prefix,
        batch_size=config.batch_size,
        run_timestamp=run_timestamp,
        sample_file=File.from_local_sync(sample_path),
        train_sample_file=File.from_local_sync(train_sample_path),
        viz_input_file=File.from_local_sync(viz_input_path),
        profile_file=File.from_local_sync(profile_path),
        upload_urls=upload_urls,
    )
    LOGGER.info("Execution URL: %s", run.url)

    LOGGER.info("Waiting for Flyte execution to complete")
    run.wait()
    if run.phase == ActionPhase.SUCCEEDED:
        LOGGER.info("Flyte execution completed successfully")
    else:
        LOGGER.info("Flyte execution finished in phase %s", run.phase)


if __name__ == "__main__":
    main()
