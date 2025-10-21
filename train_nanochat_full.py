"""
Full end-to-end nanochat training workflow with WandB logging.
This workflow trains a nanochat model on 1 GPU with complete epochs.
"""

import flyte
from datetime import datetime

# Configure the task environment with GPU and all dependencies
train_env = flyte.TaskEnvironment(
    name="nanochat-full-training",
    # Configure for 1 GPU (T4 for cost-effectiveness, can upgrade to A100:1 for speed)
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
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    num_iterations: int = 5000,  # Increased for longer training
    device_batch_size: int = 16,
    eval_every: int = 200,
) -> dict:
    """
    Complete end-to-end training: download data, train model, log to WandB.

    Everything runs in a single task to avoid container boundaries.

    Args:
        run_name: WandB run name
        depth: Model depth (8 = ~42M params)
        num_shards: Number of data shards (50 = ~2.5B tokens)
        num_iterations: Training iterations (5000 = longer training)
        device_batch_size: Batch size per device
        eval_every: Evaluate every N steps

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
    print(f"This will take 2-4 hours on T4 GPU")
    print("=" * 80 + "\n")

    # Prepare training command
    # Note: don't use -- separator, the configurator doesn't expect it
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        f"--run={run_name}",
        f"--depth={depth}",
        f"--num_iterations={num_iterations}",
        f"--device_batch_size={device_batch_size}",
        f"--total_batch_size=65536",  # Smaller batch for single GPU
        f"--max_seq_len=512",  # Smaller context for faster training
        f"--eval_every={eval_every}",
        f"--core_metric_every=2000",  # Evaluate core metrics periodically
        f"--sample_every=500",  # Sample frequently to see progress
    ]

    print("Training command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("This will take 2-4 hours. Monitor via WandB!")
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

    # Return results
    results = {
        "status": "completed",
        "run_name": run_name,
        "depth": depth,
        "num_iterations": num_iterations,
        "num_data_shards": num_shards,
        "checkpoint_dir": f"base_checkpoints/d{depth}",
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else "None",
    }

    return results


if __name__ == "__main__":
    # Initialize Flyte connection
    print("Initializing Flyte connection...")
    flyte.init_from_config(".flyte/config.yaml")

    print("\n" + "=" * 80)
    print("LAUNCHING FULL NANOCHAT TRAINING ON FLYTE")
    print("=" * 80)
    print("\nThis workflow will:")
    print("  1. Set up environment and verify GPU")
    print("  2. Clone nanochat and download FineWeb-Edu data")
    print("  3. Train nanochat model for 5000 iterations")
    print("  4. Log all metrics to WandB")
    print("\nConfiguration:")
    print("  - GPU: 1x T4 (15GB VRAM)")
    print("  - CPU: 8 cores")
    print("  - Memory: 32Gi")
    print("  - Model: 8 layers, ~42M parameters")
    print("  - Data: 50 shards (~2.5B tokens)")
    print("  - Training: 5000 iterations (~2-4 hours)")
    print("  - Logging: WandB (project: nanochat)")
    print("\n" + "=" * 80)

    # Generate unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"nanochat_d8_5k_flyte_{timestamp}"

    print(f"\nRun name: {run_name}")
    print("Submitting workflow...\n")

    # Run the workflow
    run = flyte.run(
        train_nanochat_end_to_end,
        run_name=run_name,
        depth=8,  # 8 layers = ~42M params (good for T4)
        num_shards=50,  # 50 shards = ~2.5B tokens
        num_iterations=5000,  # 5000 iterations for longer training (2-4 hours)
        device_batch_size=16,  # 16 for T4 (adjust based on GPU memory)
        eval_every=250,  # Evaluate every 250 steps
    )

    # Print run information
    print("=" * 80)
    print(f"Run ID: {run.name}")
    print(f"Run URL: {run.url}")
    print("=" * 80)
    print("\nWorkflow submitted successfully!")
    print("\nMonitor progress at:")
    print(f"  - Flyte Console: {run.url}")
    print(f"  - WandB: https://wandb.ai/wz1492/nanochat")
    print("\nEstimated time: 2-4 hours total")
    print("  - Data download: ~10 min")
    print("  - Training (5000 iterations): ~2-4 hours on T4")
    print("  - Evaluation: included during training")
    print("=" * 80)
