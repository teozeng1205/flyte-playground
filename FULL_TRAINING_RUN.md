# Full Nanochat Training Run - End to End

## Run Information

**Run Name:** nanochat_d8_flyte_20251020_002609

**Status:** Building Docker image (with git included)

**Build URL:** https://atpco.hosted.unionai.cloud/v2/runs/project/system/domain/production/rgm8bbccj9vxxhhvq9x7

**Submitted:** Oct 20, 2025 00:26 UTC

**Previous Attempt:** Failed - git was missing from Docker image (fixed)

## Workflow Overview

This is a **complete end-to-end training run** that includes:

1. ✅ Environment setup and GPU verification
2. ✅ Data download (FineWeb-Edu, 50 shards = ~2.5B tokens)
3. ✅ Full model training (1000 iterations with proper epochs)
4. ✅ WandB logging enabled (all metrics tracked)
5. ✅ Model checkpointing

## Configuration

### Hardware Resources
- **GPU:** 1x T4 (15GB VRAM)
- **CPU:** 8 cores
- **Memory:** 32Gi
- **Storage:** Ephemeral (for data and checkpoints)

### Model Configuration
- **Architecture:** GPT-style Transformer
- **Depth:** 8 layers
- **Parameters:** ~42 million
- **Context Length:** 512 tokens
- **Vocab Size:** ~50k (from tiktoken)

### Training Configuration
- **Dataset:** FineWeb-Edu (50 shards)
- **Total Tokens:** ~2.5 billion tokens
- **Iterations:** 1000
- **Batch Size (device):** 16
- **Total Batch Size:** 65,536 tokens
- **Evaluation:** Every 100 steps
- **Sampling:** Every 500 steps

### WandB Configuration
- **Project:** nanochat
- **Run Name:** nanochat_d8_flyte_20251020_001211
- **Metrics Logged:**
  - Training loss (every step)
  - Validation bits-per-byte (every 100 steps)
  - Learning rate
  - Tokens per second
  - Model FLOPs utilization (MFU)
  - Sample generations

## Workflow Stages

### Stage 1: Environment Setup
**Task:** `setup_environment()`

- Verifies Python and PyTorch versions
- Checks GPU availability and specs
- Validates WandB API key
- Reports system configuration

**Expected Duration:** 30 seconds

### Stage 2: Data Preparation
**Task:** `download_and_prepare_data(num_shards=50)`

- Clones nanochat repository
- Downloads 50 shards of FineWeb-Edu dataset
- Tokenizes data using GPT-2 tokenizer
- Creates train/val splits

**Expected Duration:** 10-15 minutes
**Data Size:** ~12.5GB raw text → ~2.5B tokens

### Stage 3: Base Model Training
**Task:** `train_base_model(...)`

- Initializes GPT model (8 layers, ~42M params)
- Trains for 1000 iterations
- Evaluates on validation set every 100 steps
- Generates samples every 500 steps
- Saves checkpoints
- Logs all metrics to WandB

**Expected Duration:** 45-75 minutes
**Estimated Throughput:** ~1000 tokens/sec on T4

### Stage 4: Results Aggregation
**Task:** `main()`

- Combines results from all stages
- Returns final metrics
- Provides WandB run URL

**Expected Duration:** < 1 minute

## Total Estimated Time

**🕐 Total: 60-90 minutes**

- Data download: 10-15 min
- Training: 45-75 min
- Overhead: 5-10 min

## Monitoring

### Flyte Console
Monitor build progress:
- https://atpco.hosted.unionai.cloud/v2/runs/project/system/domain/production/rf7nlmdcs9jxf298p5j4

Once build completes, workflow run URL will be available.

### WandB Dashboard
View training metrics in real-time:
- Project: https://wandb.ai/YOUR_USERNAME/nanochat
- Run: nanochat_d8_flyte_20251020_001211

### Key Metrics to Watch

1. **Training Loss** - Should decrease steadily
   - Initial: ~6-7 (random initialization)
   - After 1000 steps: ~3-4 (depends on data)

2. **Validation BPB** (Bits Per Byte)
   - Lower is better
   - Target: < 2.0 for decent performance

3. **Tokens/Second**
   - T4 GPU: 800-1200 tokens/sec
   - Monitor for stability

4. **MFU** (Model FLOPs Utilization)
   - Measures efficiency
   - Target: 20-40% on T4

5. **Sample Generations**
   - Quality improves over training
   - Check coherence and grammar

## What's Different from Test Run

### Previous (test run):
- ❌ No real data (random tokens)
- ❌ Only 20 iterations
- ❌ No WandB logging
- ❌ Minimal model (200K params)
- ⏱️ Runtime: ~3 minutes

### This (full run):
- ✅ Real FineWeb-Edu dataset
- ✅ 1000 iterations (proper training)
- ✅ WandB logging enabled
- ✅ Full model (42M params)
- ⏱️ Runtime: ~60-90 minutes

## Expected Outputs

### WandB Logs
```
Step 100: loss=5.234, val_bpb=2.145, lr=0.02, tok/sec=1024
Step 200: loss=4.891, val_bpb=2.089, lr=0.02, tok/sec=1031
Step 300: loss=4.623, val_bpb=2.034, lr=0.02, tok/sec=1019
...
Step 1000: loss=3.456, val_bpb=1.823, lr=0.004, tok/sec=1028
```

### Sample Generations (at step 500)
```
Prompt: "The capital of France is"
Generated: "The capital of France is Paris, located in the northern part of the country. It is known for its"

Prompt: "If yesterday was Friday, then tomorrow will be"
Generated: "If yesterday was Friday, then tomorrow will be Sunday. The days of the week follow a regular cycle"
```

### Final Results
```json
{
  "workflow": "nanochat_full_training",
  "run_name": "nanochat_d8_flyte_20251020_001211",
  "training": {
    "status": "completed",
    "depth": 8,
    "num_iterations": 1000,
    "min_validation_bpb": 1.823,
    "final_validation_bpb": 1.845,
    "checkpoint_dir": "base_checkpoints/d8"
  }
}
```

## Checkpoint Location

Checkpoints will be saved in the task's working directory:
```
/root/nanochat/base_checkpoints/d8/
├── checkpoint.pt
├── model_config.json
└── training_state.json
```

Note: These are ephemeral - to persist, you'll need to copy to persistent storage.

## Next Steps After Completion

1. **Review WandB Metrics**
   - Check training curves
   - Verify loss convergence
   - Examine sample quality

2. **Download Checkpoint** (if needed)
   - Use Flyte data management
   - Or add FlyteFile outputs to tasks

3. **Scale Up Training**
   - Increase iterations (2000, 5000, 10000)
   - Larger model (depth=12 → ~174M params)
   - More data shards (100, 200, 500)
   - Better GPU (A100 for 2-3x speedup)

4. **Fine-tuning** (next phase)
   - Add chat/instruct data
   - Supervised fine-tuning (SFT)
   - RLHF if desired

## Troubleshooting

### If Build Fails
- Check Docker image layers in build logs
- Verify pip package compatibility
- May need to pin specific versions

### If Data Download Fails
- Dataset may be temporarily unavailable
- Reduce num_shards and retry
- Check network connectivity

### If Training OOMs
- Reduce device_batch_size (16 → 8 → 4)
- Reduce max_seq_len (512 → 256)
- Or use smaller model (depth=6)

### If WandB Fails
- Check WANDB_API_KEY secret is set
- Verify wandb.ai account access
- Can continue training with offline mode

## Files

- **Workflow:** `/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/flyte/train_nanochat_full.py`
- **Documentation:** `/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/flyte/README_TRAINING.md`
- **This File:** `/Users/weichengzeng/Library/CloudStorage/OneDrive-ATPCO/Desktop/flyte/FULL_TRAINING_RUN.md`

---

**Last Updated:** Oct 20, 2025 00:12 UTC

**Status:** Building Docker image - waiting for build to complete before workflow execution starts
