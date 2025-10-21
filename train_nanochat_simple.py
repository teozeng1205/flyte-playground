"""
Simplified Flyte workflow for training nanochat.
This version uses a simpler approach that packages the nanochat code.
"""

import flyte

# Configure the task environment with GPU and all dependencies
train_env = flyte.TaskEnvironment(
    name="nanochat-gpu-training",
    # Configure for 1 GPU (using T4 as default, can change to A100:1, H100:1, etc.)
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
    image=(
        flyte.Image
        .from_debian_base((3, 12))
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
            "files-to-prompt",
        )
    ),
    env_vars={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
    },
    secrets=[flyte.Secret("WANDB_API_KEY")],
    cache=flyte.Cache("auto"),
)


@train_env.task
def check_gpu() -> dict:
    """
    Check if GPU is available and return GPU information.
    """
    import torch
    import sys

    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if hasattr(torch.version, 'cuda'):
        print("CUDA version:", torch.version.cuda)

    try:
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        print(f"GPU available: {gpu_available}")
        print(f"GPU count: {gpu_count}")

        if gpu_available and gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            print(f"GPU 0 name: {gpu_name}")
            print(f"GPU 0 memory: {gpu_memory:.2f} GB")
        else:
            gpu_name = "None"
            gpu_memory = 0
            print("WARNING: No GPU detected!")

        info = {
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
        }

        print("\nGPU Information Summary:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        return info

    except Exception as e:
        print(f"ERROR during GPU check: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_name": "Error",
            "gpu_memory_gb": 0,
            "error": str(e)
        }


@train_env.task
def train_tiny_model(config: dict) -> dict:
    """
    Train a tiny GPT model as a proof of concept.

    This is a minimal training example to test the GPU setup.
    For full nanochat training, you would need to include the nanochat source code.
    """
    import torch
    import torch.nn as nn
    import time
    import sys

    print("=" * 60)
    print("STARTING TINY MODEL TRAINING")
    print("=" * 60)
    print(f"Configuration: {config}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    try:
        # Check GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("Initializing CUDA...")
            torch.cuda.init()
            print("CUDA initialized successfully!")
    except Exception as e:
        print(f"ERROR during device setup: {str(e)}")
        import traceback
        traceback.print_exc()
        device = torch.device("cpu")
        print("Falling back to CPU")

    # Create a tiny model
    class TinyGPT(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=512,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.fc(x)

    try:
        # Initialize model
        print("\nInitializing model...")
        vocab_size = config.get("vocab_size", 1000)
        model = TinyGPT(vocab_size=vocab_size)
        print(f"Model created with vocab_size={vocab_size}")

        print(f"Moving model to device: {device}")
        model = model.to(device)
        print("Model moved to device successfully!")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Create dummy data
        batch_size = config.get("batch_size", 8)
        seq_len = config.get("seq_len", 64)
        num_iterations = config.get("num_iterations", 10)

        print(f"\nTraining config:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Iterations: {num_iterations}")

        # Optimizer
        print("\nInitializing optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print("Optimizer ready!")

        # Training loop
        print("\n" + "=" * 60)
        print("STARTING TRAINING LOOP")
        print("=" * 60)
        model.train()
        start_time = time.time()

        for step in range(num_iterations):
            # Generate random data
            x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0 or step == num_iterations - 1:
                print(f"Step {step}/{num_iterations}, Loss: {loss.item():.4f}")

            # Flush output periodically
            if step % 5 == 0:
                sys.stdout.flush()

        training_time = time.time() - start_time

        results = {
            "status": "completed",
            "num_parameters": num_params,
            "num_iterations": num_iterations,
            "final_loss": float(loss.item()),
            "training_time_seconds": float(training_time),
            "device": str(device),
        }

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Results: {results}")

        return results

    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "device": str(device),
        }


@train_env.task
def main(
    vocab_size: int = 1000,
    batch_size: int = 8,
    seq_len: int = 64,
    num_iterations: int = 20
) -> dict:
    """
    Main workflow: Check GPU and train a tiny model.

    Args:
        vocab_size: Vocabulary size
        batch_size: Batch size
        seq_len: Sequence length
        num_iterations: Number of training iterations

    Returns:
        Combined results from GPU check and training
    """
    print("=" * 60)
    print("Nanochat Training Workflow - GPU Test")
    print("=" * 60)

    # Step 1: Check GPU availability
    gpu_info = check_gpu()

    if not gpu_info["gpu_available"]:
        print("WARNING: No GPU available! Training will be slow.")

    # Step 2: Train model
    config = {
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iterations": num_iterations,
    }

    training_results = train_tiny_model(config)

    # Combine results
    final_results = {
        "gpu_info": gpu_info,
        "training": training_results,
    }

    print("\n" + "=" * 60)
    print("Workflow Completed Successfully!")
    print("=" * 60)

    return final_results


if __name__ == "__main__":
    # Initialize Flyte connection
    print("Initializing Flyte connection...")
    flyte.init_from_config(".flyte/config.yaml")

    print("\n" + "=" * 60)
    print("Launching Nanochat GPU Training Workflow")
    print("=" * 60)
    print("\nThis is a test workflow that will:")
    print("1. Check GPU availability")
    print("2. Train a tiny transformer model on the GPU")
    print("\nConfiguration:")
    print("  - GPU: 1x (configured in TaskEnvironment)")
    print("  - Model: Tiny GPT (2 layers, 4 heads)")
    print("  - Training: 20 iterations")
    print("\n" + "=" * 60)

    # Run the workflow
    run = flyte.run(
        main,
        vocab_size=1000,
        batch_size=8,
        seq_len=64,
        num_iterations=20,
    )

    # Print run information
    print(f"\nRun ID: {run.name}")
    print(f"Run URL: {run.url}")
    print("\nWorkflow submitted! Monitor progress at the URL above.")
    print("\nTo wait for completion, uncomment the run.wait() line in the script.")

    # Uncomment to wait for completion:
    # print("\nWaiting for workflow to complete...")
    # result = run.wait()
    # print(f"\nFinal Results: {result}")
