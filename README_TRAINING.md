# Nanochat Training with Flyte

This directory contains Flyte workflows for training a tiny version of nanochat using GPU resources.

## Files Created

### 1. `train_nanochat_simple.py` (Recommended)
A simplified GPU training workflow that:
- Checks GPU availability
- Trains a tiny transformer model (proof of concept)
- Uses 1 GPU (T4 by default, configurable to A100, H100, etc.)
- Demonstrates the Flyte GPU workflow pattern

**Configuration:**
- GPU: T4:1 (can be changed to A100:1, H100:1, etc. in the code)
- CPU: 4 cores
- Memory: 16Gi
- Model: Tiny GPT (2 layers, 4 heads, ~200K parameters)
- Training: 20 iterations (fast test run)

### 2. `train_nanochat.py`
A more complete workflow designed for full nanochat training that:
- Downloads and prepares training data
- Trains a base model using nanochat's training scripts
- Requires the nanochat source code to be available in the container

**Note:** This requires additional setup to include the nanochat source code in the Docker image.

### 3. `example.py`
A basic Flyte workflow example (fixed to work correctly) that demonstrates:
- Simple task definitions
- Task chaining
- Remote execution

## Quick Start

### Run the Simple GPU Training Test

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the simplified training workflow
python train_nanochat_simple.py
```

This will:
1. Build a Docker image with PyTorch and dependencies
2. Submit the workflow to Flyte
3. Allocate a T4 GPU
4. Run GPU check and training tasks
5. Print the run URL for monitoring

### Monitor the Run

The script will output a URL like:
```
Run URL: https://atpco.hosted.unionai.cloud/v2/runs/project/flytesnacks/domain/development/<run-id>
```

Visit this URL to monitor:
- Real-time logs
- GPU utilization
- Training progress
- Task completion status

## Current Status

**The training workflow is currently building the Docker image.**

The image build includes:
- Python 3.12 (Debian base)
- PyTorch (with CUDA support)
- Training dependencies (wandb, datasets, tiktoken, etc.)

This can take 5-10 minutes on the first run. Subsequent runs will use the cached image.

### To check the build progress:
Visit: https://atpco.hosted.unionai.cloud/v2/runs/project/system/domain/production/r2fjxxqqprrbptwjp7c2

## Configuration Options

### Changing GPU Type

In `train_nanochat_simple.py` (line 12), you can change:

```python
resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1")
```

To use different GPUs:
- `gpu="A100:1"` - Single A100 GPU
- `gpu="H100:1"` - Single H100 GPU
- `gpu="L4:1"` - Single L4 GPU
- `gpu="A100 80G:1"` - Single A100 80GB GPU

### Adjusting Training Parameters

In the `if __name__ == "__main__":` section, modify:

```python
run = flyte.run(
    main,
    vocab_size=1000,      # Vocabulary size
    batch_size=8,         # Batch size
    seq_len=64,          # Sequence length
    num_iterations=20,   # Number of training iterations
)
```

## Full Nanochat Training

To train the full nanochat model, you'll need to:

1. Package the nanochat source code into the Docker image
2. Use `train_nanochat.py` instead
3. Adjust resources based on model size:
   - Tiny model (depth=6): ~42M params, T4 GPU, 8GB memory
   - Small model (depth=12): ~174M params, A100 GPU, 16GB memory
   - Medium model (depth=20): ~480M params, A100 80GB GPU, 32GB memory

## Monitoring GPU Usage

The workflow includes a `check_gpu()` task that reports:
- GPU availability
- GPU count
- GPU name/model
- GPU memory capacity

This information is logged in the task outputs.

## Next Steps

1. **Wait for the current build to complete** (monitor the build URL)
2. **Check the run results** at the run URL
3. **Verify GPU allocation and training completion**
4. **Adjust parameters** as needed for your use case
5. **Scale up** to full nanochat training once the test workflow succeeds

## Troubleshooting

### Image Build Takes Long
- First builds can take 5-10 minutes
- PyTorch installation is the largest component
- Subsequent builds use cached layers

### GPU Not Available
- Check that GPU resource is correctly specified
- Verify your Flyte cluster has GPU nodes
- Check quota limits in your organization

### Training Fails
- Check logs at the run URL
- Verify memory allocation is sufficient
- Adjust batch_size or model size for available GPU memory

## Files Structure

```
flyte/
├── .flyte/
│   └── config.yaml              # Flyte connection config
├── nanochat/                    # Nanochat source code
│   ├── nanochat/               # Python package
│   ├── scripts/                # Training scripts
│   └── pyproject.toml          # Dependencies
├── example.py                   # Basic Flyte example (fixed)
├── train_nanochat_simple.py    # Simple GPU training test (RECOMMENDED)
├── train_nanochat.py           # Full nanochat training workflow
└── README_TRAINING.md          # This file
```

## Resources

- Flyte Docs: https://www.union.ai/docs/v1/flyte/user-guide/
- Nanochat: https://github.com/karpathy/nanochat
- Your Flyte Console: https://atpco.hosted.unionai.cloud

## Notes

- The workflow uses WANDB_MODE=offline to avoid requiring API keys during testing
- GPU resources are specified using the format "TYPE:COUNT" (e.g., "T4:1")
- The .venv/ directory contains the local Python environment
- All remote execution happens in containerized environments built from the Image specifications
