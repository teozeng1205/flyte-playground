"""
Full end-to-end nanochat training workflow with WandB logging.
This workflow trains a nanochat model on 1 GPU with complete epochs.
"""

import flyte
from datetime import datetime

# Configure the task environment with GPU and all dependencies
train_env = flyte.TaskEnvironment(
    name="nanochat-full-training",
    resources=flyte.Resources(cpu=8, memory="32Gi", gpu=1),
    image=(
        flyte.Image
        .from_debian_base((3, 12))
        .with_apt_packages(
            "git",  # For cloning nanochat repo
            "curl",  # For downloading Rust installer
            "build-essential",  # For building Rust packages
        )
        .with_pip_packages(
            "torch",
            "datasets",
            "wandb",
            "tiktoken",
            "numpy",
            "regex",
            "tokenizers",
            "psutil",
            "fastapi",
            "uvicorn",
            "maturin",  # Required for building rustbpe
        )
    ),
    env_vars={
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    },
    secrets=[flyte.Secret("WANDB_API_KEY")],
    cache=flyte.Cache("auto"),
)


@train_env.task
def train_nanochat_end_to_end(
    run_name: str,
    depth: int = 8,
    num_shards: int = 50,
    num_iterations: int = 5000,  # Reduced for faster iterations
    device_batch_size: int = 16,
    eval_every: int = 200,
    base_total_batch_size: int = 65536,
    base_max_seq_len: int = 512,
    base_core_metric_every: int = 2000,
    base_sample_every: int = 500,
    base_loss_device_batch_size: int = 4,
    base_loss_split_tokens: int = 262144,
    mid_device_batch_size: int = 4,
    mid_total_batch_size: int = 32768,
    mid_init_lr_frac: float = 0.5,
    mid_eval_every: int = 100,
    mid_eval_num_samples: int = 1,
    mid_eval_batch_size: int = 4,
    mid_eval_max_new_tokens: int = 256,
    mid_eval_task: str | None = None,
    sft_target_examples_per_step: int = 32,
    sft_eval_every: int = 100,
    sft_eval_steps: int = 100,
    sft_eval_num_samples: int = 1,
    sft_eval_batch_size: int = 4,
    sft_eval_max_new_tokens: int = 256,
    sft_eval_task: str | None = None,
    rl_examples_per_step: int = 8,
    rl_num_samples: int = 4,
    rl_eval_examples: int = 200,
    rl_save_every: int = 60,
    rl_eval_every: int = 60,
    rl_resume: bool = True,
    rl_eval_num_samples: int = 2,
    rl_eval_batch_size: int = 4,
    rl_eval_max_new_tokens: int = 256,
    rl_eval_task: str = "GSM8K",
) -> dict:
    """
    Complete end-to-end training: download data, train model, log to WandB.

    Everything runs in a single task to avoid container boundaries.

    Args:
        run_name: WandB run name
        depth: Model depth (8 = ~42M params)
        num_shards: Number of data shards (50 = ~2.5B tokens)
        num_iterations: Training iterations (1000 = shorter training)
        device_batch_size: Batch size per device
        eval_every: Evaluate every N steps
        base_total_batch_size: Total tokens per optimizer step during base training
        base_max_seq_len: Context length used for base training
        base_core_metric_every: Frequency (steps) for CORE metric evaluation during base training
        base_sample_every: Sampling frequency (steps) during base training
        base_loss_device_batch_size: Per-device batch size for loss evaluation
        base_loss_split_tokens: Tokens per split when computing base loss
        mid_device_batch_size: Per-device batch size for mid-training
        mid_total_batch_size: Total tokens per optimizer step during mid-training
        mid_init_lr_frac: Mid-training initial LR multiplier
        mid_eval_every: Evaluation cadence (steps) during mid-training
        mid_eval_num_samples: Samples per prompt for mid-stage chat evaluation
        mid_eval_batch_size: Batch size for categorical mid-stage evaluation
        mid_eval_max_new_tokens: Max generated tokens for mid-stage chat evaluation
        mid_eval_task: Optional task list override for mid-stage chat evaluation
        sft_target_examples_per_step: Target examples per step during SFT
        sft_eval_every: Evaluation cadence (steps) during SFT
        sft_eval_steps: Number of evaluation steps during SFT
        sft_eval_num_samples: Samples per prompt for SFT chat evaluation
        sft_eval_batch_size: Batch size for categorical SFT evaluation
        sft_eval_max_new_tokens: Max generated tokens for SFT chat evaluation
        sft_eval_task: Optional task list override for SFT chat evaluation
        rl_examples_per_step: GSM8K examples per RL optimization step
        rl_num_samples: Samples to draw per RL prompt
        rl_eval_examples: GSM8K examples used during RL evaluation
        rl_save_every: Checkpoint save cadence (minutes) during RL
        rl_eval_every: Evaluation cadence (minutes) during RL
        rl_resume: Whether to resume RL from latest checkpoint
        rl_eval_num_samples: Samples per prompt for RL chat evaluation
        rl_eval_batch_size: Batch size for categorical RL evaluation
        rl_eval_max_new_tokens: Max generated tokens for RL chat evaluation
        rl_eval_task: Task name used for RL chat evaluation

    Returns:
        Training results and metrics
    """
    import subprocess
    import os
    import sys
    import torch

    print("=" * 80)
    print("NANOCHAT END-TO-END TRAINING")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Model depth: {depth} (~{42 * (depth/8):.0f}M parameters)")
    print(f"Data: {num_shards} shards (~{num_shards * 0.05:.1f}B tokens)")
    print(f"Training: {num_iterations} iterations")
    print(f"Device batch size: {device_batch_size}")
    print("=" * 80 + "\n")

    # ========================================================================
    # STAGE 1: Environment Check
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: ENVIRONMENT CHECK")
    print("=" * 80)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")

    if gpu_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Check WandB
    try:
        import wandb
        print(f"WandB version: {wandb.__version__}")
        wandb_api_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_api_key:
            print(f"WandB API key found (length: {len(wandb_api_key)})")
        else:
            print("WARNING: WandB API key not found!")
    except Exception as e:
        print(f"WandB check failed: {e}")

    # ========================================================================
    # STAGE 2: Clone Nanochat and Download Data
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 2: DOWNLOADING DATA ({num_shards} shards)")
    print("=" * 80)

    # Clone nanochat repo
    print("\nCloning nanochat repository...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/karpathy/nanochat.git"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Git clone output: {result.stdout}")
        print(f"Git clone error: {result.stderr}")
        raise RuntimeError(f"Git clone failed: {result.stderr}")
    print("Repository cloned successfully!")

    # Change to nanochat directory
    os.chdir("nanochat")
    nanochat_dir = os.getcwd()
    print(f"Working directory: {nanochat_dir}")

    stage_results = {}

    def run_stage(stage_key: str, description: str, command: list[str], extra_env: dict | None = None):
        """
        Execute a subprocess command for a given stage, mirroring the sequential workflow in speedrun.sh.
        """
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)
        print("Command:", " ".join(command))
        sys.stdout.flush()

        cmd_env = os.environ.copy()
        if extra_env:
            cmd_env.update(extra_env)

        result = subprocess.run(
            command,
            cwd=nanochat_dir,
            env=cmd_env,
            text=True,
        )

        if result.returncode != 0:
            stage_results[stage_key] = {
                "status": "failed",
                "returncode": result.returncode,
                "command": command,
            }
            raise RuntimeError(f"{stage_key} failed with return code {result.returncode}")

        stage_results[stage_key] = {
            "status": "completed",
            "command": command,
        }
    # ========================================================================
    # Install Rust and build rustbpe module
    # ========================================================================
    print("\n" + "=" * 80)
    print("INSTALLING RUST AND BUILDING RUSTBPE MODULE")
    print("=" * 80)

    # Check if Rust is already installed
    rust_installed = False
    try:
        rust_check = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if rust_check.returncode == 0:
            rust_installed = True
            print(f"\nRust already installed: {rust_check.stdout.strip()}")
    except FileNotFoundError:
        pass  # Rust not installed, will install below

    if not rust_installed:
        print("\nInstalling Rust...")
        # Install Rust using rustup (non-interactive)
        rust_install = subprocess.run(
            ["sh", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"],
            capture_output=True,
            text=True,
            shell=False
        )

        print(rust_install.stdout)
        if rust_install.stderr:
            print("STDERR:", rust_install.stderr)

        if rust_install.returncode != 0:
            raise RuntimeError(f"Rust installation failed: {rust_install.stderr}")

        # Update PATH to include cargo
        cargo_bin = os.path.expanduser("~/.cargo/bin")
        if os.path.exists(cargo_bin):
            os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
            print(f"Updated PATH to include: {cargo_bin}")

        # Verify installation
        verify = subprocess.run(
            [os.path.expanduser("~/.cargo/bin/rustc"), "--version"],
            capture_output=True,
            text=True
        )
        print(f"Rust installed successfully: {verify.stdout.strip()}")

    # Build and install the nanochat package (which includes rustbpe)
    print("\nBuilding nanochat package with rustbpe module...")

    # Use maturin to build the Rust extension into a wheel
    print("Building Rust extension with maturin...")
    result = subprocess.run(
        [sys.executable, "-m", "maturin", "build", "--release", "--out", "/tmp/wheels"],
        capture_output=True,
        text=True,
        cwd=nanochat_dir,
        env={**os.environ, "PATH": os.environ.get("PATH", "")}
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build rustbpe wheel: {result.stderr}")

    # Find the built wheel
    import glob
    wheels = glob.glob("/tmp/wheels/*.whl")
    if not wheels:
        raise RuntimeError("No wheel file found after maturin build")

    wheel_path = wheels[0]
    print(f"Built wheel: {wheel_path}")

    # Install the wheel directly using Python's zipfile module
    # Extract to a temporary directory first, then copy to nanochat directory
    print("Installing rustbpe module...")
    import zipfile
    import shutil

    # Extract to temporary directory
    temp_extract = "/tmp/wheel_extract"
    os.makedirs(temp_extract, exist_ok=True)

    print(f"Extracting wheel to {temp_extract}...")
    with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
        # Extract all files except metadata directories
        for member in zip_ref.namelist():
            # Skip .dist-info metadata but keep all actual code
            if '.dist-info/' not in member:
                print(f"  Extracting: {member}")
                zip_ref.extract(member, temp_extract)

    # Copy rustbpe to nanochat directory where it will be importable
    rustbpe_src = os.path.join(temp_extract, "rustbpe")
    rustbpe_dst = os.path.join(nanochat_dir, "rustbpe")

    if os.path.exists(rustbpe_src):
        if os.path.exists(rustbpe_dst):
            shutil.rmtree(rustbpe_dst)
        shutil.copytree(rustbpe_src, rustbpe_dst)
        print(f"Copied rustbpe module to {rustbpe_dst}")

    # Also add nanochat directory to Python path so rustbpe can be imported
    if nanochat_dir not in sys.path:
        sys.path.insert(0, nanochat_dir)
        print(f"Added {nanochat_dir} to Python path")

    print("Nanochat package built and installed successfully!")

    # Verify the rustbpe module is importable
    verify_env = os.environ.copy()
    verify_env["PYTHONPATH"] = nanochat_dir
    verify_import = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, '" + nanochat_dir + "'); import rustbpe; print('rustbpe module imported successfully'); print(f'rustbpe location: {rustbpe.__file__}')"],
        capture_output=True,
        text=True,
        env=verify_env
    )
    print(verify_import.stdout)
    if verify_import.returncode != 0:
        print(f"Warning: Could not import rustbpe: {verify_import.stderr}")
        # List what's in nanochat directory to debug
        print(f"\nContents of {nanochat_dir}:")
        for item in os.listdir(nanochat_dir)[:20]:
            print(f"  {item}")

    print("=" * 80)

    # Download data
    print(f"\nDownloading {num_shards} data shards...")
    print("This will take several minutes...")

    result = subprocess.run(
        [sys.executable, "-m", "nanochat.dataset", "-n", str(num_shards)],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Data download failed: {result.stderr}")

    # Verify data was downloaded
    cache_dir = os.path.expanduser("~/.cache/nanochat/base_data")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
        print(f"\nData downloaded successfully!")
        print(f"Cache directory: {cache_dir}")
        print(f"Downloaded {len(cache_files)} parquet files")
    else:
        raise RuntimeError(f"Data download failed - cache directory not found")

    # ========================================================================
    # STAGE 2.6: Download evaluation bundle (mirrors speedrun.sh)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2.6: FETCHING CORE EVALUATION BUNDLE")
    print("=" * 80)

    eval_bundle_url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
    eval_bundle_dir = os.path.expanduser("~/.cache/nanochat/eval_bundle")

    if os.path.isdir(eval_bundle_dir):
        print(f"Evaluation bundle already present at {eval_bundle_dir}")
    else:
        print("Downloading evaluation bundle (required for CORE metrics)...")
        import urllib.request
        import zipfile
        import io
        import shutil

        os.makedirs(os.path.dirname(eval_bundle_dir), exist_ok=True)
        with urllib.request.urlopen(eval_bundle_url) as response:
            archive_bytes = response.read()
        print(f"Downloaded {len(archive_bytes) / (1024 * 1024):.2f} MB, extracting...")
        temp_extract_dir = "/tmp/eval_bundle_extract"
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        os.makedirs(temp_extract_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        extracted_path = os.path.join(temp_extract_dir, "eval_bundle")
        if not os.path.isdir(extracted_path):
            raise RuntimeError("Expected eval_bundle directory not found in archive")
        if os.path.exists(eval_bundle_dir):
            shutil.rmtree(eval_bundle_dir)
        shutil.move(extracted_path, eval_bundle_dir)
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        print(f"Evaluation bundle available at {eval_bundle_dir}")

    try:
        run_stage(
            "report_reset",
            "STAGE 2.7: Resetting Nanochat report",
            [sys.executable, "-m", "nanochat.report", "reset"],
        )
    except RuntimeError as reset_err:
        # files-to-prompt sometimes times out in generate_header; fall back to a lightweight header
        error_msg = str(reset_err)
        if "report_reset failed" not in error_msg:
            raise
        print("Report reset via nanochat.report failed; generating minimal header manually.")
        try:
            from nanochat import report as report_mod
            from nanochat.common import get_base_dir as _get_base_dir
        except Exception as import_err:
            raise RuntimeError(f"Unable to import nanochat report module for fallback header: {import_err}") from reset_err

        base_cache_dir = _get_base_dir()
        report_dir = os.path.join(base_cache_dir, "report")
        os.makedirs(report_dir, exist_ok=True)

        # Clear expected files and existing report/header if present
        expected_files = getattr(report_mod, "EXPECTED_FILES", [])
        for file_name in expected_files:
            file_path = os.path.join(report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        for filename in ("report.md", "header.md"):
            path = os.path.join(report_dir, filename)
            if os.path.exists(path):
                os.remove(path)

        try:
            # Try original header generation, but guard against failures
            header_text = report_mod.generate_header()
        except Exception as header_err:
            print(f"Original generate_header failed ({header_err}); writing simplified header.")
            header_text = (
                "# nanochat training report\n\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "## Environment\n"
                "- Header generation fallback used (files-to-prompt unavailable or timed out)\n\n"
                "### Bloat\n"
                "- Characters: unavailable\n"
                "- Lines: unavailable\n"
                "- Files: unavailable\n"
                "- Tokens (approx): unavailable\n"
                "- Dependencies (uv.lock lines): unavailable\n\n"
            )

        header_path = os.path.join(report_dir, "header.md")
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_path, "w") as f:
            f.write(header_text)
            f.write(f"Run started: {start_time}\n\n---\n\n")

        stage_results["report_reset"] = {
            "status": "completed_with_fallback",
            "header_path": header_path,
            "error": error_msg,
        }
        print(f"Fallback report header written to {header_path}")

    # ========================================================================
    # STAGE 2.5: Build Tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2.5: BUILDING TOKENIZER")
    print("=" * 80)

    # The tokenizer needs to be built/trained before training
    print("\nBuilding tokenizer...")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.tok_train"],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("WARNING: Tokenizer training failed, but tokenizer might already exist")
        # Check if tokenizer exists
        tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
        if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
            raise RuntimeError(f"Tokenizer build failed and tokenizer not found: {result.stderr}")

    print("Tokenizer ready!")

    # ========================================================================
    # STAGE 3: Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 3: TRAINING MODEL")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Depth: {depth}")
    print(f"Iterations: {num_iterations}")
    print(f"This will take ~1 hour on T4 GPU")
    print("=" * 80 + "\n")

    # Prepare training command
    # Note: don't use -- separator, the configurator doesn't expect it
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        f"--run={run_name}",
        f"--depth={depth}",
        f"--num_iterations={num_iterations}",
        f"--device_batch_size={device_batch_size}",
        f"--total_batch_size={base_total_batch_size}",  # Smaller batch for single GPU
        f"--max_seq_len={base_max_seq_len}",  # Smaller context for faster training
        f"--eval_every={eval_every}",
        f"--core_metric_every={base_core_metric_every}",  # Evaluate core metrics periodically
        f"--sample_every={base_sample_every}",  # Sample frequently to see progress
    ]

    print("Training command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("This will take ~1 hour. Monitor via WandB!")
    print("=" * 80 + "\n")

    # Run training with real-time output streaming
    # Don't capture output so it streams to stdout in real-time
    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output in real-time
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Training completed successfully
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Iterations completed: {num_iterations}")
    print(f"Checkpoint directory: base_checkpoints/d{depth}")
    print("\nView full results in WandB:")
    print(f"  https://wandb.ai/wz1492/nanochat/runs/{run_name}")
    print("=" * 80 + "\n")

    stage_results["base_train"] = {
        "status": "completed",
        "command": cmd,
        "run_name": run_name,
        "num_iterations": num_iterations,
        "checkpoint_dir": f"base_checkpoints/d{depth}",
        "wandb_run": run_name,
    }

    # STAGE 4: Base loss & sampling
    base_loss_cmd = [
        sys.executable,
        "-m",
        "scripts.base_loss",
        f"--device_batch_size={base_loss_device_batch_size}",  # Reduced from 8 to avoid OOM on T4
        f"--split_tokens={base_loss_split_tokens}",
    ]
    run_stage(
        "base_loss",
        "STAGE 4: Base loss evaluation & sampling",
        base_loss_cmd,
    )

    # STAGE 5: CORE evaluation for base model
    base_eval_cmd = [sys.executable, "-m", "scripts.base_eval"]
    run_stage(
        "base_eval",
        "STAGE 5: CORE evaluation (base model)",
        base_eval_cmd,
    )

    # STAGE 6: Midtraining
    mid_run_name = f"{run_name}_mid"
    mid_train_cmd = [
        sys.executable,
        "-m",
        "scripts.mid_train",
        f"--run={mid_run_name}",
        f"--device_batch_size={mid_device_batch_size}",  # Reduced from 8 to avoid OOM on T4
        f"--total_batch_size={mid_total_batch_size}",  # Reduced proportionally to maintain gradient accumulation
        f"--init_lr_frac={mid_init_lr_frac}",
        f"--eval_every={mid_eval_every}",
    ]
    run_stage(
        "mid_train",
        "STAGE 6: Midtraining (tool use & conversational blend)",
        mid_train_cmd,
    )
    stage_results["mid_train"]["run_name"] = mid_run_name

    # STAGE 7: Chat evaluation for mid model
    mid_eval_cmd = [
        sys.executable,
        "-m",
        "scripts.chat_eval",
        "-i",
        "mid",
        "-n",
        str(mid_eval_num_samples),
        "-b",
        str(mid_eval_batch_size),
        "-m",
        str(mid_eval_max_new_tokens),
    ]
    if mid_eval_task:
        mid_eval_cmd.extend(["-a", mid_eval_task])
    run_stage(
        "mid_eval",
        "STAGE 7: Chat evaluation (mid model)",
        mid_eval_cmd,
    )
    stage_results["mid_eval"]["source"] = "mid"

    # STAGE 8: Supervised finetuning (SFT)
    sft_run_name = f"{run_name}_sft"
    sft_cmd = [
        sys.executable,
        "-m",
        "scripts.chat_sft",
        f"--run={sft_run_name}",
        "--source=mid",
        f"--target_examples_per_step={sft_target_examples_per_step}",
        f"--eval_every={sft_eval_every}",
        f"--eval_steps={sft_eval_steps}",
    ]
    run_stage(
        "chat_sft",
        "STAGE 8: Supervised finetuning (SFT)",
        sft_cmd,
    )
    stage_results["chat_sft"]["run_name"] = sft_run_name

    # STAGE 9: Chat evaluation for SFT model
    sft_eval_cmd = [
        sys.executable,
        "-m",
        "scripts.chat_eval",
        "-i",
        "sft",
        "-n",
        str(sft_eval_num_samples),
        "-b",
        str(sft_eval_batch_size),
        "-m",
        str(sft_eval_max_new_tokens),
    ]
    if sft_eval_task:
        sft_eval_cmd.extend(["-a", sft_eval_task])
    run_stage(
        "sft_eval",
        "STAGE 9: Chat evaluation (SFT model)",
        sft_eval_cmd,
    )
    stage_results["sft_eval"]["source"] = "sft"

    # STAGE 10: Reinforcement learning on GSM8K
    rl_run_name = f"{run_name}_rl"
    rl_cmd = [
        sys.executable,
        "-m",
        "scripts.chat_rl",
        f"--run={rl_run_name}",
        "--source=sft",
        f"--examples_per_step={rl_examples_per_step}",
        f"--num_samples={rl_num_samples}",
        f"--eval_examples={rl_eval_examples}",
        f"--save_every={rl_save_every}",
        f"--eval_every={rl_eval_every}",
    ]
    if not rl_resume:
        rl_cmd.append("--resume=False")
    run_stage(
        "chat_rl",
        "STAGE 10: Reinforcement learning (GSM8K)",
        rl_cmd,
    )
    stage_results["chat_rl"]["run_name"] = rl_run_name
    stage_results["chat_rl"]["source"] = "sft"

    # STAGE 11: Chat evaluation for RL model
    rl_eval_cmd = [
        sys.executable,
        "-m",
        "scripts.chat_eval",
        "-i",
        "rl",
        "-n",
        str(rl_eval_num_samples),
        "-b",
        str(rl_eval_batch_size),
        "-m",
        str(rl_eval_max_new_tokens),
    ]
    if rl_eval_task:
        rl_eval_cmd.extend(["-a", rl_eval_task])
    run_stage(
        "rl_eval",
        "STAGE 11: Chat evaluation (RL model - GSM8K)",
        rl_eval_cmd,
    )
    stage_results["rl_eval"]["source"] = "rl"
    stage_results["rl_eval"]["task"] = rl_eval_task or "all"

    # STAGE 12: Generate final report
    run_stage(
        "report_generate",
        "STAGE 12: Generating final nanochat report",
        [sys.executable, "-m", "nanochat.report", "generate"],
    )
    base_cache_dir = os.path.expanduser("~/.cache/nanochat")
    stage_results["report_generate"]["report_path"] = os.path.join(base_cache_dir, "report", "report.md")
    stage_results["report_generate"]["working_copy"] = os.path.join(nanochat_dir, "report.md")

    final_results = {
        "status": "completed",
        "run_name": run_name,
        "depth": depth,
        "num_iterations": num_iterations,
        "num_data_shards": num_shards,
        "checkpoint_dir": f"base_checkpoints/d{depth}",
        "wandb": {
            "base": run_name,
            "mid": mid_run_name,
            "sft": sft_run_name,
            "rl": rl_run_name,
        },
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else "None",
        "stages": stage_results,
    }

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit nanochat end-to-end training to Flyte")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name / job name prefix")
    parser.add_argument("--depth", type=int, default=4, help="Model depth (smaller = faster). Default: 4")
    parser.add_argument("--num_shards", type=int, default=5, help="Number of dataset shards to download. Default: 5")
    parser.add_argument("--num_iterations", type=int, default=500, help="Base training iterations. Default: 500")
    parser.add_argument("--device_batch_size", type=int, default=8, help="Per-device batch size. Default: 8")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N steps. Default: 100")

    base_group = parser.add_argument_group("base training", "Parameters for base training and loss evaluation")
    base_group.add_argument("--base-total-batch-size", type=int, default=65536, help="Total batch size (tokens) for base training.")
    base_group.add_argument("--base-max-seq-len", type=int, default=512, help="Max sequence length used for base training.")
    base_group.add_argument("--base-core-metric-every", type=int, default=2000, help="Frequency (steps) for CORE metric evaluation during base training.")
    base_group.add_argument("--base-sample-every", type=int, default=500, help="Frequency (steps) for sampling during base training.")
    base_group.add_argument("--base-loss-device-batch-size", type=int, default=4, help="Per-device batch size for base loss evaluation.")
    base_group.add_argument("--base-loss-split-tokens", type=int, default=262144, help="Tokens per split for base loss evaluation.")

    mid_group = parser.add_argument_group("mid training", "Parameters for the mid-training stage and evaluation")
    mid_group.add_argument("--mid-device-batch-size", type=int, default=4, help="Per-device batch size for mid training.")
    mid_group.add_argument("--mid-total-batch-size", type=int, default=32768, help="Total batch size (tokens) for mid training.")
    mid_group.add_argument("--mid-init-lr-frac", type=float, default=0.5, help="Initial learning rate fraction for mid training.")
    mid_group.add_argument("--mid-eval-every", type=int, default=100, help="Evaluation cadence (steps) during mid training.")
    mid_group.add_argument("--mid-eval-num-samples", type=int, default=1, help="Samples per prompt for mid-stage chat evaluation.")
    mid_group.add_argument("--mid-eval-batch-size", type=int, default=4, help="Batch size for categorical mid-stage evaluation.")
    mid_group.add_argument("--mid-eval-max-new-tokens", type=int, default=256, help="Max new tokens for mid-stage chat evaluation.")
    mid_group.add_argument("--mid-eval-task", type=str, default=None, help="Optional override for mid-stage evaluation task list.")

    sft_group = parser.add_argument_group("sft training", "Parameters for SFT training and evaluation")
    sft_group.add_argument("--sft-target-examples-per-step", type=int, default=32, help="Target examples per optimizer step during SFT.")
    sft_group.add_argument("--sft-eval-every", type=int, default=100, help="Evaluation cadence (steps) during SFT.")
    sft_group.add_argument("--sft-eval-steps", type=int, default=100, help="Number of evaluation steps during SFT.")
    sft_group.add_argument("--sft-eval-num-samples", type=int, default=1, help="Samples per prompt for SFT chat evaluation.")
    sft_group.add_argument("--sft-eval-batch-size", type=int, default=4, help="Batch size for categorical SFT evaluation.")
    sft_group.add_argument("--sft-eval-max-new-tokens", type=int, default=256, help="Max new tokens for SFT chat evaluation.")
    sft_group.add_argument("--sft-eval-task", type=str, default=None, help="Optional override for SFT evaluation task list.")

    rl_group = parser.add_argument_group("rl stage", "Parameters for RL training and evaluation")
    rl_group.add_argument("--rl-examples-per-step", type=int, default=8, help="Examples per optimization step during RL.")
    rl_group.add_argument("--rl-num-samples", type=int, default=4, help="Samples per prompt during RL training.")
    rl_group.add_argument("--rl-eval-examples", type=int, default=200, help="Examples evaluated during RL.")
    rl_group.add_argument("--rl-save-every", type=int, default=60, help="Minutes between RL checkpoints.")
    rl_group.add_argument("--rl-eval-every", type=int, default=60, help="Minutes between RL evaluations.")
    rl_group.add_argument("--rl-resume", action=argparse.BooleanOptionalAction, default=True, help="Resume RL from the latest checkpoint.")
    rl_group.add_argument("--rl-eval-num-samples", type=int, default=2, help="Samples per prompt during RL evaluation.")
    rl_group.add_argument("--rl-eval-batch-size", type=int, default=4, help="Batch size for categorical RL evaluation.")
    rl_group.add_argument("--rl-eval-max-new-tokens", type=int, default=256, help="Max new tokens for RL chat evaluation.")
    rl_group.add_argument("--rl-eval-task", type=str, default="GSM8K", help="Task name for RL evaluation (default: GSM8K).")

    args = parser.parse_args()

    # Initialize Flyte connection
    print("Initializing Flyte connection...")
    flyte.init_from_config(".flyte/config.yaml")

    # Generate unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"nanochat_smoke_d{args.depth}_{args.num_iterations}_{timestamp}"
    print(f"\nRun name: {run_name}")
    print("Submitting workflow...\n")

    # Run the workflow
    task_kwargs = vars(args).copy()
    task_kwargs["run_name"] = run_name
    run = flyte.run(
        train_nanochat_end_to_end,
        **task_kwargs,
    )
