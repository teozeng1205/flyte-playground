# dco_visualize

Flyte workflow for sampling DCO parquet data, training a foundation-style tabular encoder, segmenting the resulting row embeddings, and publishing a travel-native dashboard to S3.

## Default target

- Source day prefix: `s3://s3-atp-3victors-3vprod-use1-derived-common-output/v1/AA/2026/03/07/`
- Output: `s3://3v-teo-dev/dco_visualize/`

The workflow samples recursively across all hourly parquet shards under the requested day, trains on a bounded stratified subset, runs batched inference over the target partition, and renders a standalone dashboard with fare calendar, metro flow, market matrix, segment fingerprints, and densMAP diagnostics.

## Submission

Run from the repo root:

```bash
.venv/bin/python dco_visualize/submit.py \
  --customer AA \
  --sales-date 2026-03-07 \
  --sample-rows 100000 \
  --train-rows 100000 \
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
  --embedding-dims 32 \
  --output-prefix s3://3v-teo-dev/dco_visualize/ \
  --auth-type DeviceFlow
```

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
- `embedding_density.png`
- `metro_flow_map.png`
- `fare_calendar.png`
- `market_matrix.png`
- `segment_fingerprint.png`
- `manifest.json`
