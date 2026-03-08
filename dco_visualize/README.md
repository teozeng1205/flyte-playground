# dco_visualize

Flyte workflow for sampling DCO parquet data, fitting two TabPFN 2.5 branches over the raw DCO columns, extracting row embeddings directly from the fitted regressors, and publishing a comparison dashboard to S3.

## Default target

- Source day prefix: `s3://s3-atp-3victors-3vprod-use1-derived-common-output/v1/AA/2026/03/07/`
- Output: `s3://3v-teo-dev/dco_visualize/`

The workflow samples recursively across all hourly parquet shards under the requested day, trains on a bounded subset, runs batched inference over the target partition, and renders a standalone dashboard with:

- pretrained TabPFN 2.5 embeddings
- fine-tuned TabPFN 2.5 embeddings
- route network and market matrix views
- fare calendar surfaces
- segment fingerprints and branch-agreement diagnostics

The Flyte task name stays `dco-visualize-overwatch.execute`.

## Submission

Run from the repo root:

```bash
.venv/bin/python dco_visualize/submit.py \
  --customer AA \
  --sales-date 2026-03-07 \
  --sample-rows 100000 \
  --train-rows 50000 \
  --viz-rows 50000 \
  --output-prefix s3://3v-teo-dev/dco_visualize/
```

If the browser-based PKCE flow is blocked in the terminal session, use device flow:

```bash
.venv/bin/python dco_visualize/submit.py \
  --customer AA \
  --sales-date 2026-03-07 \
  --sample-rows 1000 \
  --train-rows 1000 \
  --viz-rows 500 \
  --output-prefix s3://3v-teo-dev/dco_visualize/ \
  --auth-type DeviceFlow
```

## Local demo

Run a bounded local DCO demo, modeled after the upstream TabPFN notebook flow but specialized to AA/day DCO data:

```bash
.venv/bin/python dco_visualize/tabpfn_dco_demo_local.py \
  --customer AA \
  --sales-date 2026-03-07 \
  --sample-rows 5000 \
  --train-rows 2500 \
  --viz-rows 2000
```

This writes local artifacts under `dco_visualize/demo_outputs/`.

## Artifacts

Each run publishes:

- `profile.json`
- `sample.parquet`
- `embeddings_full.parquet`
- `viz_sample.parquet`
- `market_aggregates.parquet`
- `embedding_bundle.pt`
- `metrics.json`
- `dashboard.html`
- `pretrained_embedding_density.png`
- `finetuned_embedding_density.png`
- `route_network.png`
- `fare_calendar.png`
- `market_matrix.png`
- `segment_fingerprint.png`
- `manifest.json`
