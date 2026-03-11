from __future__ import annotations

import ast
import argparse
import logging
import os
import re
import subprocess
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
    sample_parquet_files,
    write_json,
)
from dco_qwen3_visualize.sampling import build_representative_samples_from_parquet
from dco_qwen3_visualize.workflow import execute, render_artifacts

LOGGER = logging.getLogger(__name__)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ARTIFACT_FILENAMES = [
    "profile.json",
    "sample.parquet",
    "context_sample.parquet",
    "eval_sample.parquet",
    "viz_sample.parquet",
    "pretrained_embeddings.parquet",
    "finetuned_embeddings.parquet",
    "finetune_pairs.jsonl",
    "finetuned_adapter.tar.gz",
    "training_stats.png",
    "metrics.json",
    "dashboard.html",
    "manifest.json",
]


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
    parser.add_argument("--upload-url-expiry-hours", type=int, default=48)
    parser.add_argument("--recover-render-run", default="")
    parser.add_argument("--recover-render-action", default="")
    return parser.parse_args()


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _extract_balanced_literal(text: str, opener: str) -> str:
    closer = {"{": "}", "[": "]"}[opener]
    start = text.find(opener)
    if start < 0:
        raise ValueError(f"Could not find opener {opener!r}")

    depth = 0
    in_string = False
    string_char = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            string_char = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"Could not extract balanced literal starting with {opener!r}")


def _collapse_newlines_inside_strings(text: str) -> str:
    pieces: list[str] = []
    in_string = False
    string_char = ""
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                pieces.append(char)
                escaped = False
                continue
            if char == "\\":
                pieces.append(char)
                escaped = True
                continue
            if char == string_char:
                in_string = False
                string_char = ""
                pieces.append(char)
                continue
            if char in {"\n", "\r"}:
                continue
            pieces.append(char)
            continue
        if char in {"'", '"'}:
            in_string = True
            string_char = char
        pieces.append(char)
    return "".join(pieces)


def _run_flyte_cli(*args: str) -> str:
    command = [
        str(REPO_ROOT / "dco_visualize" / ".venv" / "bin" / "flyte"),
        "-c",
        str(REPO_ROOT / ".flyte" / "config.yaml"),
        "--org",
        "atpco",
        "--output-format",
        "json",
        *args,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return _strip_ansi(completed.stdout)


def _get_recovery_render_inputs(run_name: str, action_name: str | None = None) -> dict[str, object]:
    actions_output = _run_flyte_cli("get", "action", run_name, "-p", "flytesnacks", "-d", "development")
    actions = ast.literal_eval(_collapse_newlines_inside_strings(_extract_balanced_literal(actions_output, "[")))
    render_action = action_name
    if not render_action:
        for action in actions:
            task_name = str(action.get("metadata", {}).get("task", {}).get("id", {}).get("name", ""))
            if task_name.endswith("render_artifacts"):
                render_action = str(action["id"]["name"])
                break
    if not render_action:
        raise ValueError(f"Could not find render_artifacts action for run {run_name}")

    io_output = _run_flyte_cli(
        "get",
        "io",
        run_name,
        render_action,
        "-p",
        "flytesnacks",
        "-d",
        "development",
        "--inputs-only",
    )
    return ast.literal_eval(_collapse_newlines_inside_strings(_extract_balanced_literal(io_output, "{")))


def _init_flyte(auth_type: str) -> None:
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
        auth_type=auth_type,
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


def _recover_render_only(args: argparse.Namespace) -> None:
    recovery_inputs = _get_recovery_render_inputs(
        run_name=args.recover_render_run,
        action_name=args.recover_render_action or None,
    )
    config = DCOQwen3VisualizeConfig(
        customer=str(recovery_inputs["customer"]),
        sales_date=str(recovery_inputs["sales_date"]),
        sample_rows=int(recovery_inputs["sample_rows"]),
        train_rows=int(recovery_inputs["train_rows"]),
        viz_rows=int(recovery_inputs["viz_rows"]),
        output_prefix=str(recovery_inputs["output_prefix"]),
        batch_size=int(recovery_inputs["batch_size"]),
    )
    config.validate()
    run_timestamp = str(recovery_inputs["run_timestamp"])
    destination_prefix = config.run_output_prefix(run_timestamp)
    upload_urls = generate_presigned_upload_urls(
        destination_prefix=destination_prefix,
        filenames=ARTIFACT_FILENAMES,
        profile_name=args.output_aws_profile,
        expires_in=args.upload_url_expiry_hours * 60 * 60,
    )

    _init_flyte(args.auth_type)
    run = flyte.with_runcontext(copy_style="all").run(
        render_artifacts,
        sample_file=File.from_existing_remote(str(recovery_inputs["sample_file"])),
        context_sample_file=File.from_existing_remote(str(recovery_inputs["context_sample_file"])),
        eval_sample_file=File.from_existing_remote(str(recovery_inputs["eval_sample_file"])),
        profile_file=File.from_existing_remote(str(recovery_inputs["profile_file"])),
        viz_sample_file=File.from_existing_remote(str(recovery_inputs["viz_sample_file"])),
        pretrained_embeddings_file=File.from_existing_remote(str(recovery_inputs["pretrained_embeddings_file"])),
        finetuned_embeddings_file=File.from_existing_remote(str(recovery_inputs["finetuned_embeddings_file"])),
        finetune_pairs_file=File.from_existing_remote(str(recovery_inputs["finetune_pairs_file"])),
        adapter_tar_file=File.from_existing_remote(str(recovery_inputs["adapter_tar_file"])),
        training_stats_file=File.from_existing_remote(str(recovery_inputs["training_stats_file"])),
        metrics_file=File.from_existing_remote(str(recovery_inputs["metrics_file"])),
        customer=config.customer,
        sales_date=config.sales_date,
        sample_rows=config.sample_rows,
        train_rows=config.train_rows,
        viz_rows=config.viz_rows,
        output_prefix=config.output_prefix,
        batch_size=config.batch_size,
        run_timestamp=run_timestamp,
        upload_urls=upload_urls,
    )
    LOGGER.info("Render-only recovery execution URL: %s", run.url)
    run.wait()
    LOGGER.info("Render-only recovery finished in phase %s", run.phase)


def main() -> None:
    _configure_logging()
    args = parse_args()
    if args.recover_render_run:
        _recover_render_only(args)
        return

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

    _init_flyte(args.auth_type)

    run_timestamp = make_run_timestamp()
    destination_prefix = config.run_output_prefix(run_timestamp)
    upload_urls = generate_presigned_upload_urls(
        destination_prefix=destination_prefix,
        filenames=ARTIFACT_FILENAMES,
        profile_name=args.output_aws_profile,
        expires_in=args.upload_url_expiry_hours * 60 * 60,
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
