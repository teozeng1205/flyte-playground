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

from dco_qwen3_visualize.config import DCOQwen3VisualizeConfig, make_run_timestamp
from dco_qwen3_visualize.io import (
    collect_parquet_metadata,
    generate_presigned_upload_urls,
    list_s3_parquet_objects,
    materialize_parquet_files,
    sample_parquet_files,
    write_json,
)
from dco_qwen3_visualize.sampling import build_representative_samples_from_parquet
from dco_qwen3_visualize.workflow import execute

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the DCO Qwen3 visualization Flyte workflow.")
    parser.add_argument("--customer", default="AA")
    parser.add_argument("--sales-date", default=date.today().isoformat())
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--viz-rows", type=int, default=50_000)
    parser.add_argument("--output-prefix", default="s3://3v-teo-dev/dco_visualize/")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--auth-type", default="Pkce", choices=["Pkce", "DeviceFlow"])
    parser.add_argument("--source-aws-profile", default="3VPROD")
    parser.add_argument("--output-aws-profile", default="3VDEV")
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    config = DCOQwen3VisualizeConfig(
        customer=args.customer,
        sales_date=args.sales_date,
        sample_rows=args.sample_rows,
        train_rows=args.train_rows,
        viz_rows=args.viz_rows,
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
    sample_rows = min(config.sample_rows, total_rows)
    train_rows = min(config.train_rows, sample_rows, 50_000)
    viz_rows = min(config.viz_rows, sample_rows)
    config = DCOQwen3VisualizeConfig(
        customer=args.customer,
        sales_date=args.sales_date,
        sample_rows=sample_rows,
        train_rows=train_rows,
        viz_rows=viz_rows,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
    )
    config.validate()

    staging_dir = Path(tempfile.mkdtemp(prefix="dco_qwen3_submit_"))
    sample_path = staging_dir / "sample.parquet"
    context_path = staging_dir / "context_sample.parquet"
    eval_path = staging_dir / "eval_sample.parquet"
    profile_path = staging_dir / "profile.json"
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
    representative = build_representative_samples_from_parquet(
        sample_path,
        train_rows=config.train_rows,
        viz_rows=config.viz_rows,
        random_seed=config.random_seed,
        batch_size=config.batch_size,
        config=config,
        train_output_path=context_path,
        viz_output_path=eval_path,
    )
    profile["representative_sampling"] = representative
    write_json(profile_path, profile)

    config_path = REPO_ROOT / ".flyte" / "config.yaml"
    flyte_config = Config.auto(config_path)
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
    upload_urls = generate_presigned_upload_urls(
        destination_prefix=destination_prefix,
        filenames=[
            "profile.json",
            "sample.parquet",
            "context_sample.parquet",
            "eval_sample.parquet",
            "viz_sample.parquet",
            "pretrained_embeddings.parquet",
            "finetuned_embeddings.parquet",
            "finetune_pairs.jsonl",
            "finetuned_adapter.tar.gz",
            "metrics.json",
            "dashboard.html",
            "manifest.json",
        ],
        profile_name=args.output_aws_profile,
    )

    run = flyte.with_runcontext(copy_style="all").run(
        execute,
        customer=config.customer,
        sales_date=config.sales_date,
        sample_rows=config.sample_rows,
        train_rows=config.train_rows,
        viz_rows=config.viz_rows,
        output_prefix=config.output_prefix,
        batch_size=config.batch_size,
        run_timestamp=run_timestamp,
        sample_file=File.from_local_sync(sample_path),
        context_sample_file=File.from_local_sync(context_path),
        eval_sample_file=File.from_local_sync(eval_path),
        profile_file=File.from_local_sync(profile_path),
        upload_urls=upload_urls,
    )
    LOGGER.info("Execution URL: %s", run.url)
    run.wait()
    if run.phase == ActionPhase.SUCCEEDED:
        LOGGER.info("Flyte execution completed successfully")
    else:
        LOGGER.info("Flyte execution finished in phase %s", run.phase)


if __name__ == "__main__":
    main()
