# DCO Qwen3 Visualize

Flyte workflow for DCO row serialization, Qwen3 embedding inference, LoRA
fine-tuning, and standalone pretrained-versus-fine-tuned dashboard generation.

The workflow keeps the same DCO sampling and publication pattern used by the
existing TabPFN visualization package, but replaces the model core with:

- `Qwen/Qwen3-Embedding-0.6B` pretrained row embeddings
- LoRA fine-tuning on mined DCO similarity pairs
- 2D projection and side-by-side manifold comparison

Outputs are published under:

`s3://3v-teo-dev/dco_visualize/qwen3_v1/customer=<customer>/sales_date=<date>/run_ts=<ts>/`
