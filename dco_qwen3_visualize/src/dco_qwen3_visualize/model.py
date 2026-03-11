from __future__ import annotations

import gc
import json
import logging
import math
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
import umap

from dco_qwen3_visualize.config import DCOQwen3VisualizeConfig
from dco_qwen3_visualize.io import METADATA_COLUMNS, write_json
from dco_qwen3_visualize.progress import format_duration, progress_snapshot

LOGGER = logging.getLogger(__name__)

CORE_FIELD_ORDER = [
    "origin",
    "destination",
    "origin_city",
    "destination_city",
    "origin_metro",
    "destination_metro",
    "origin_country",
    "destination_country",
    "trip_type",
    "cabin",
    "stops",
    "carrier",
    "source",
    "advance_purchase",
    "outbound_departure_date",
    "inbound_departure_date",
    "price_inc",
    "price_exc",
    "tax",
]

DISPLAY_COLUMNS = [
    "carrier",
    "source",
    "trip_type",
    "cabin",
    "stops",
    "origin_metro",
    "destination_metro",
    "origin",
    "destination",
    "outbound_departure_date",
    "inbound_departure_date",
    "advance_purchase",
    "price_inc",
]


@dataclass(frozen=True)
class ProjectionResult:
    coordinates: np.ndarray
    trustworthiness: float
    fit_rows: int
    trust_rows: int
    method: str


@dataclass(frozen=True)
class QwenEmbeddingEncoder:
    tokenizer: Any
    model: Any
    device: str
    config: DCOQwen3VisualizeConfig

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        arrays: list[np.ndarray] = []
        effective_batch_size = batch_size or self.config.inference_batch_size
        started_at = time.perf_counter()
        self.model.eval()
        for offset in range(0, len(texts), effective_batch_size):
            batch = texts[offset : offset + effective_batch_size]
            with torch.no_grad():
                embeddings = _encode_batch_tensor(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=batch,
                    config=self.config,
                    device=self.device,
                    normalize=True,
                )
            arrays.append(embeddings.cpu().numpy())
            if offset == 0 or offset + effective_batch_size >= len(texts) or ((offset // effective_batch_size) + 1) % self.config.progress_log_every_batches == 0:
                snapshot = progress_snapshot(min(offset + len(batch), len(texts)), len(texts), started_at)
                LOGGER.info(
                    "Embedding progress: rows=%d/%d pct=%.1f elapsed=%s rate=%.0f rows/s remaining=%s eta_utc=%s",
                    snapshot.done,
                    snapshot.total,
                    snapshot.percent,
                    format_duration(snapshot.elapsed_seconds),
                    snapshot.rate_per_second,
                    format_duration(snapshot.remaining_seconds),
                    snapshot.eta_utc or "unknown",
                )
        if not arrays:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        return np.concatenate(arrays, axis=0)


def _device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ordered_feature_columns(columns: list[str]) -> list[str]:
    ordered = [column for column in CORE_FIELD_ORDER if column in columns]
    ordered.extend(sorted(column for column in columns if column not in set(ordered)))
    return ordered


def _display_columns(columns: list[str]) -> list[str]:
    ordered = [column for column in DISPLAY_COLUMNS if column in columns]
    ordered.extend(column for column in columns if column not in ordered and column in {"pos", "channel", "currency"})
    return ordered[:12]


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return "missing"
    return str(value)


def serialize_dco_row(row: pd.Series | dict[str, object], feature_columns: list[str]) -> str:
    if isinstance(row, pd.Series):
        values = row.to_dict()
    else:
        values = row
    ordered_columns = _ordered_feature_columns(feature_columns)
    lines = ["airfare offer row"]
    for column in ordered_columns:
        lines.append(f"{column}: {_format_value(values.get(column))}")
    return "\n".join(lines)


def serialize_dco_frame(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    ordered_columns = _ordered_feature_columns(feature_columns)
    texts: list[str] = []
    for row in frame[ordered_columns].to_dict(orient="records"):
        texts.append(serialize_dco_row(row, ordered_columns))
    return texts


def _extract_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in set(METADATA_COLUMNS)]


def _parse_departure_dates(frame: pd.DataFrame) -> pd.Series:
    column = "outbound_departure_date" if "outbound_departure_date" in frame.columns else None
    if column is None:
        return pd.Series([pd.NaT] * len(frame), index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce")


def _prepare_pair_mining_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    for column in [
        "origin_metro",
        "destination_metro",
        "trip_type",
        "cabin",
        "stops",
        "carrier",
        "source",
        "origin",
        "destination",
    ]:
        if column in working.columns:
            working[column] = working[column].astype("string").fillna("missing")
        else:
            working[column] = "missing"
    working["metro_od"] = working["origin_metro"] + "->" + working["destination_metro"]
    working["strict_key"] = (
        working["metro_od"]
        + "|"
        + working["trip_type"]
        + "|"
        + working["cabin"]
        + "|"
        + working["stops"]
    )
    working["price_inc_numeric"] = pd.to_numeric(working.get("price_inc"), errors="coerce")
    working["departure_date"] = _parse_departure_dates(working)
    working["row_hash"] = pd.util.hash_pandas_object(working["row_id"].astype("string"), index=False).astype(np.uint64)
    return working


def build_similarity_pairs(frame: pd.DataFrame, texts: list[str], config: DCOQwen3VisualizeConfig) -> list[dict[str, object]]:
    working = _prepare_pair_mining_frame(frame)
    strict_groups = {
        key: group.index.to_numpy(dtype=np.int64)
        for key, group in working.groupby("strict_key", observed=False)
    }
    metro_groups = {
        key: group.index.to_numpy(dtype=np.int64)
        for key, group in working.groupby("metro_od", observed=False)
    }
    all_indices = np.arange(len(working), dtype=np.int64)

    candidate_anchor_indices = working.sort_values(["row_hash", "price_inc_numeric"], kind="mergesort").index.to_numpy(dtype=np.int64)
    candidate_anchor_indices = candidate_anchor_indices[: min(config.finetune_pair_rows, len(candidate_anchor_indices))]
    pair_records: list[dict[str, object]] = []

    for anchor_index in candidate_anchor_indices:
        anchor = working.iloc[anchor_index]
        anchor_price = anchor["price_inc_numeric"]
        anchor_departure = anchor["departure_date"]

        strict_candidates = strict_groups.get(anchor["strict_key"], np.empty((0,), dtype=np.int64))
        strict_candidates = strict_candidates[strict_candidates != anchor_index]
        positive_candidates = []
        for candidate_index in strict_candidates.tolist():
            candidate = working.iloc[candidate_index]
            candidate_price = candidate["price_inc_numeric"]
            if pd.notna(anchor_price) and pd.notna(candidate_price):
                tolerance = max(abs(float(anchor_price)), 1.0) * config.finetune_positive_price_tolerance
                if abs(float(candidate_price) - float(anchor_price)) > tolerance:
                    continue
            if pd.notna(anchor_departure) and pd.notna(candidate["departure_date"]):
                if abs((candidate["departure_date"] - anchor_departure).days) > config.finetune_positive_day_tolerance:
                    continue
            positive_candidates.append(candidate_index)

        if not positive_candidates:
            metro_candidates = metro_groups.get(anchor["metro_od"], np.empty((0,), dtype=np.int64))
            metro_candidates = metro_candidates[metro_candidates != anchor_index]
            for candidate_index in metro_candidates.tolist():
                candidate = working.iloc[candidate_index]
                if candidate["trip_type"] != anchor["trip_type"]:
                    continue
                if pd.notna(anchor_price) and pd.notna(candidate["price_inc_numeric"]):
                    tolerance = max(abs(float(anchor_price)), 1.0) * (config.finetune_positive_price_tolerance * 1.5)
                    if abs(float(candidate["price_inc_numeric"]) - float(anchor_price)) > tolerance:
                        continue
                positive_candidates.append(candidate_index)

        if not positive_candidates:
            metro_candidates = metro_groups.get(anchor["metro_od"], np.empty((0,), dtype=np.int64))
            metro_candidates = metro_candidates[metro_candidates != anchor_index]
            positive_candidates.extend(int(candidate_index) for candidate_index in metro_candidates.tolist())

        if not positive_candidates:
            continue

        positive_index = int(
            min(
                positive_candidates,
                key=lambda candidate_index: (
                    abs(float(working.iloc[candidate_index]["price_inc_numeric"] or 0.0) - float(anchor_price or 0.0)),
                    abs(
                        int((working.iloc[candidate_index]["departure_date"] - anchor_departure).days)
                        if pd.notna(anchor_departure) and pd.notna(working.iloc[candidate_index]["departure_date"])
                        else 10_000
                    ),
                    int(working.iloc[candidate_index]["row_hash"]),
                ),
            )
        )

        negative_indices: list[int] = []
        same_market_candidates = metro_groups.get(anchor["metro_od"], np.empty((0,), dtype=np.int64))
        for candidate_index in same_market_candidates.tolist():
            if candidate_index in {anchor_index, positive_index}:
                continue
            candidate = working.iloc[candidate_index]
            candidate_price = candidate["price_inc_numeric"]
            price_is_far = True
            if pd.notna(anchor_price) and pd.notna(candidate_price):
                tolerance = max(abs(float(anchor_price)), 1.0) * (config.finetune_positive_price_tolerance * 2.0)
                price_is_far = abs(float(candidate_price) - float(anchor_price)) > tolerance
            if price_is_far or candidate["cabin"] != anchor["cabin"] or candidate["stops"] != anchor["stops"]:
                negative_indices.append(int(candidate_index))
            if len(negative_indices) >= config.finetune_max_negatives:
                break

        if len(negative_indices) < config.finetune_max_negatives:
            cross_market_candidates = all_indices[
                (~np.isin(all_indices, [anchor_index, positive_index]))
                & (working["metro_od"].to_numpy() != anchor["metro_od"])
            ]
            for candidate_index in cross_market_candidates[: config.finetune_max_negatives - len(negative_indices)]:
                negative_indices.append(int(candidate_index))

        if len(negative_indices) < config.finetune_max_negatives:
            continue

        pair_records.append(
            {
                "anchor_row_id": str(anchor["row_id"]),
                "positive_row_id": str(working.iloc[positive_index]["row_id"]),
                "negative_row_ids": [str(working.iloc[index]["row_id"]) for index in negative_indices[: config.finetune_max_negatives]],
                "anchor_text": texts[anchor_index],
                "positive_text": texts[positive_index],
                "negative_texts": [texts[index] for index in negative_indices[: config.finetune_max_negatives]],
                "strict_key": str(anchor["strict_key"]),
                "metro_od": str(anchor["metro_od"]),
            }
        )
    return pair_records


def _ensure_hf_auth() -> None:
    if not os.environ.get("HF_TOKEN"):
        LOGGER.warning("HF_TOKEN is not set; gated Qwen model download may fail")
    else:
        LOGGER.info("Using HF_TOKEN from environment for Qwen runtime")


def _format_query(text: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {text}"


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths]


def _truncate_and_normalize_embeddings(tensor: torch.Tensor, embedding_dim: int, *, normalize: bool) -> torch.Tensor:
    if tensor.size(1) > embedding_dim:
        tensor = tensor[:, :embedding_dim]
    if normalize:
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    return tensor


def _load_qwen_encoder(
    config: DCOQwen3VisualizeConfig,
    *,
    adapter_path: str | Path | None = None,
    trainable: bool = False,
) -> QwenEmbeddingEncoder:
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    device = _device_string()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    try:
        model = AutoModel.from_pretrained(config.model_id, attn_implementation="flash_attention_2", **model_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(config.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    model.to(device)
    if trainable:
        model.train()
    else:
        model.eval()
    return QwenEmbeddingEncoder(tokenizer=tokenizer, model=model, device=device, config=config)


def _encode_batch_tensor(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: DCOQwen3VisualizeConfig,
    device: str,
    *,
    normalize: bool,
) -> torch.Tensor:
    formatted = [_format_query(text, config.row_instruction) for text in texts]
    tokens = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=config.sequence_max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    outputs = model(**tokens)
    pooled = _last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
    return _truncate_and_normalize_embeddings(pooled, config.embedding_dim, normalize=normalize)


def _build_lora_model(base_encoder: QwenEmbeddingEncoder, config: DCOQwen3VisualizeConfig) -> Any:
    from peft import LoraConfig, get_peft_model

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base_encoder.model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def _fine_tune_qwen3_adapter(
    pair_records: list[dict[str, object]],
    config: DCOQwen3VisualizeConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pair_records:
        write_json(output_dir / "adapter_metrics.json", {"status": "skipped", "reason": "no_pairs"})
        return {"status": "skipped", "reason": "no_pairs"}

    started_at = time.perf_counter()
    base_encoder = _load_qwen_encoder(config, trainable=True)
    model = _build_lora_model(base_encoder, config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.finetune_learning_rate,
        weight_decay=config.finetune_weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    device = base_encoder.device
    batch_size = config.finetune_batch_size
    losses: list[float] = []
    global_step = 0

    for epoch in range(config.finetune_epochs):
        epoch_records = pair_records[: min(len(pair_records), config.finetune_max_steps_per_epoch * batch_size)]
        for batch_start in range(0, len(epoch_records), batch_size):
            batch_records = epoch_records[batch_start : batch_start + batch_size]
            anchors = [str(record["anchor_text"]) for record in batch_records]
            positives = [str(record["positive_text"]) for record in batch_records]
            negatives_nested = [list(record["negative_texts"])[: config.finetune_max_negatives] for record in batch_records]
            negatives = [text for group in negatives_nested for text in group]

            anchor_embeddings = _encode_batch_tensor(
                model=model,
                tokenizer=base_encoder.tokenizer,
                texts=anchors,
                config=config,
                device=device,
                normalize=True,
            )
            positive_embeddings = _encode_batch_tensor(
                model=model,
                tokenizer=base_encoder.tokenizer,
                texts=positives,
                config=config,
                device=device,
                normalize=True,
            )
            negative_embeddings = _encode_batch_tensor(
                model=model,
                tokenizer=base_encoder.tokenizer,
                texts=negatives,
                config=config,
                device=device,
                normalize=True,
            ).reshape(len(batch_records), config.finetune_max_negatives, -1)

            positive_scores = (anchor_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
            negative_scores = torch.einsum("bd,bnd->bn", anchor_embeddings, negative_embeddings)
            logits = torch.cat([positive_scores, negative_scores], dim=1) / config.finetune_temperature
            labels = torch.zeros(len(batch_records), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            scaled_loss = loss / config.finetune_gradient_accumulation_steps
            scaled_loss.backward()

            if (global_step + 1) % config.finetune_gradient_accumulation_steps == 0 or batch_start + batch_size >= len(epoch_records):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu().item()))
            global_step += 1
            if global_step <= 2 or global_step % config.progress_log_every_batches == 0:
                LOGGER.info(
                    "Fine-tune progress: epoch=%d step=%d loss=%.4f elapsed=%s",
                    epoch + 1,
                    global_step,
                    losses[-1],
                    format_duration(time.perf_counter() - started_at),
                )

    model.save_pretrained(output_dir)
    base_encoder.tokenizer.save_pretrained(output_dir)
    metrics = {
        "status": "succeeded",
        "epochs": config.finetune_epochs,
        "steps": global_step,
        "pairs": len(pair_records),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "final_loss": losses[-1] if losses else None,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    write_json(output_dir / "adapter_metrics.json", metrics)
    del model
    del base_encoder
    _cleanup_memory()
    return metrics


def _tar_directory(source_dir: str | Path, output_path: str | Path) -> None:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(source_dir, arcname=source_dir.name)


def _fit_segment_ids(embeddings: np.ndarray, config: DCOQwen3VisualizeConfig) -> tuple[np.ndarray, dict[str, object]]:
    if len(embeddings) < 2:
        return np.zeros((len(embeddings),), dtype=np.int64), {"kind": "constant", "segment_count": 1}
    clusters = min(config.segment_clusters, max(2, len(embeddings) // 250))
    model = MiniBatchKMeans(n_clusters=clusters, random_state=config.random_seed, batch_size=2_048)
    labels = model.fit_predict(embeddings).astype(np.int64)
    return labels, {"kind": "minibatch_kmeans", "segment_count": int(clusters)}


def _fit_layout(embeddings: np.ndarray, config: DCOQwen3VisualizeConfig) -> ProjectionResult:
    if len(embeddings) == 0:
        return ProjectionResult(
            coordinates=np.zeros((0, 2), dtype=np.float32),
            trustworthiness=float("nan"),
            fit_rows=0,
            trust_rows=0,
            method="umap_transform",
        )
    fit_rows = min(config.layout_fit_rows, len(embeddings))
    trust_rows = min(config.trustworthiness_rows, len(embeddings))
    fit_embeddings = embeddings[:fit_rows]
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=30,
        min_dist=0.05,
        random_state=config.random_seed,
        low_memory=True,
    )
    if len(embeddings) <= fit_rows:
        coordinates = reducer.fit_transform(embeddings).astype(np.float32)
    else:
        reducer.fit(fit_embeddings)
        coordinates = reducer.transform(embeddings).astype(np.float32)
    trust_score = float(
        trustworthiness(
            embeddings[:trust_rows],
            coordinates[:trust_rows],
            n_neighbors=min(10, max(2, trust_rows - 1)),
        )
    )
    return ProjectionResult(
        coordinates=coordinates,
        trustworthiness=trust_score,
        fit_rows=fit_rows,
        trust_rows=trust_rows,
        method="umap_transform",
    )


def _prediction_diagnostics(embeddings: np.ndarray, prices: pd.Series) -> dict[str, object]:
    max_rows = min(2_000, len(embeddings))
    if len(embeddings) > max_rows:
        embeddings = embeddings[:max_rows]
        prices = prices.iloc[:max_rows]
    valid = pd.to_numeric(prices, errors="coerce").dropna()
    if len(valid) < 2 or len(embeddings) < 2:
        return {"price_neighbor_correlation": None}
    neighbor_count = min(8, len(embeddings) - 1)
    distances = pairwise_distances(embeddings, metric="cosine")
    nearest = np.argsort(distances, axis=1)[:, 1 : neighbor_count + 1]
    aligned_prices = pd.to_numeric(prices, errors="coerce").to_numpy(dtype=np.float64)
    neighbor_means = np.nanmean(aligned_prices[nearest], axis=1)
    correlation = np.corrcoef(aligned_prices, neighbor_means)[0, 1]
    return {"price_neighbor_correlation": float(correlation) if np.isfinite(correlation) else None}


def _embedding_frame(base_frame: pd.DataFrame, embeddings: np.ndarray, prefix: str) -> pd.DataFrame:
    data = {
        f"{prefix}_emb_{index:03d}": embeddings[:, index]
        for index in range(embeddings.shape[1])
    }
    return pd.concat(
        [base_frame.reset_index(drop=True), pd.DataFrame(data, index=base_frame.index)],
        axis=1,
    )


def _build_viz_frame(
    eval_frame: pd.DataFrame,
    feature_columns: list[str],
    pretrained_embeddings: np.ndarray,
    finetuned_embeddings: np.ndarray,
    pretrained_segments: np.ndarray,
    finetuned_segments: np.ndarray,
    pretrained_layout: ProjectionResult,
    finetuned_layout: ProjectionResult,
) -> pd.DataFrame:
    display_columns = [column for column in _display_columns(feature_columns) if column in eval_frame.columns]
    frame = eval_frame[METADATA_COLUMNS + display_columns].copy().reset_index(drop=True)
    pretrained_frame = _embedding_frame(frame[METADATA_COLUMNS], pretrained_embeddings, "pretrained").drop(columns=METADATA_COLUMNS)
    finetuned_frame = _embedding_frame(frame[METADATA_COLUMNS], finetuned_embeddings, "finetuned").drop(columns=METADATA_COLUMNS)
    viz_frame = pd.concat([frame, pretrained_frame, finetuned_frame], axis=1)
    viz_frame["pretrained_layout_x"] = pretrained_layout.coordinates[:, 0]
    viz_frame["pretrained_layout_y"] = pretrained_layout.coordinates[:, 1]
    viz_frame["finetuned_layout_x"] = finetuned_layout.coordinates[:, 0]
    viz_frame["finetuned_layout_y"] = finetuned_layout.coordinates[:, 1]
    viz_frame["pretrained_segment_id"] = pretrained_segments
    viz_frame["finetuned_segment_id"] = finetuned_segments
    viz_frame["layout_method"] = pretrained_layout.method
    return viz_frame


def run_qwen3_visualization(
    context_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    config: DCOQwen3VisualizeConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    _ensure_hf_auth()
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = _extract_feature_columns(context_frame)
    hover_columns = _display_columns(feature_columns)
    LOGGER.info(
        "Preparing Qwen3 visualization run: context_rows=%d eval_rows=%d features=%d model=%s",
        len(context_frame),
        len(eval_frame),
        len(feature_columns),
        config.model_id,
    )

    context_texts = serialize_dco_frame(context_frame, feature_columns)
    eval_texts = serialize_dco_frame(eval_frame, feature_columns)
    pair_records = build_similarity_pairs(context_frame, context_texts, config)
    finetune_pairs_path = output_dir / "finetune_pairs.jsonl"
    with finetune_pairs_path.open("w", encoding="utf-8") as handle:
        for record in pair_records:
            handle.write(json.dumps(record) + "\n")
    LOGGER.info("Built %d similarity pairs for LoRA fine-tuning", len(pair_records))

    pretrained_encoder = _load_qwen_encoder(config)
    pretrained_eval_embeddings = pretrained_encoder.encode(eval_texts, batch_size=config.inference_batch_size)
    pretrained_segments, pretrained_segment_metrics = _fit_segment_ids(pretrained_eval_embeddings, config)
    pretrained_layout = _fit_layout(pretrained_eval_embeddings, config)
    pretrained_metrics = {
        "model_id": config.model_id,
        "device": pretrained_encoder.device,
        "embedding_dim": int(pretrained_eval_embeddings.shape[1]),
        "pair_count": int(len(pair_records)),
        "projection_trustworthiness": pretrained_layout.trustworthiness,
        "projection_fit_rows": pretrained_layout.fit_rows,
        "projection_trust_rows": pretrained_layout.trust_rows,
        **pretrained_segment_metrics,
        **_prediction_diagnostics(pretrained_eval_embeddings, eval_frame.get(config.target_column, pd.Series(dtype=float))),
    }
    LOGGER.info(
        "Pretrained Qwen branch complete: embedding_dim=%d trust=%.3f elapsed=%s",
        pretrained_eval_embeddings.shape[1],
        pretrained_layout.trustworthiness,
        format_duration(time.perf_counter() - started_at),
    )
    del pretrained_encoder
    _cleanup_memory()

    adapter_dir = output_dir / "qwen3_lora_adapter"
    finetune_metrics = _fine_tune_qwen3_adapter(pair_records, config, adapter_dir)
    finetuned_encoder = _load_qwen_encoder(config, adapter_path=adapter_dir)
    finetuned_eval_embeddings = finetuned_encoder.encode(eval_texts, batch_size=config.inference_batch_size)
    finetuned_segments, finetuned_segment_metrics = _fit_segment_ids(finetuned_eval_embeddings, config)
    finetuned_layout = _fit_layout(finetuned_eval_embeddings, config)
    finetuned_metrics = {
        "model_id": config.model_id,
        "device": finetuned_encoder.device,
        "embedding_dim": int(finetuned_eval_embeddings.shape[1]),
        "projection_trustworthiness": finetuned_layout.trustworthiness,
        "projection_fit_rows": finetuned_layout.fit_rows,
        "projection_trust_rows": finetuned_layout.trust_rows,
        **finetuned_segment_metrics,
        **_prediction_diagnostics(finetuned_eval_embeddings, eval_frame.get(config.target_column, pd.Series(dtype=float))),
        "finetune": finetune_metrics,
    }
    del finetuned_encoder
    _cleanup_memory()

    viz_frame = _build_viz_frame(
        eval_frame=eval_frame,
        feature_columns=feature_columns,
        pretrained_embeddings=pretrained_eval_embeddings,
        finetuned_embeddings=finetuned_eval_embeddings,
        pretrained_segments=pretrained_segments,
        finetuned_segments=finetuned_segments,
        pretrained_layout=pretrained_layout,
        finetuned_layout=finetuned_layout,
    )
    pretrained_frame = _embedding_frame(eval_frame[METADATA_COLUMNS], pretrained_eval_embeddings, "pretrained")
    finetuned_frame = _embedding_frame(eval_frame[METADATA_COLUMNS], finetuned_eval_embeddings, "finetuned")

    adapter_tar_path = output_dir / "finetuned_adapter.tar.gz"
    _tar_directory(adapter_dir, adapter_tar_path)
    metrics = {
        "encoder_backend": "qwen3_embedding",
        "model_id": config.model_id,
        "row_instruction": config.row_instruction,
        "feature_columns": feature_columns,
        "hover_columns": hover_columns,
        "display_columns": _display_columns(feature_columns),
        "pair_count": int(len(pair_records)),
        "train_rows": int(len(context_frame)),
        "viz_rows": int(len(eval_frame)),
        "pretrained": pretrained_metrics,
        "finetuned": finetuned_metrics,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    LOGGER.info(
        "Completed Qwen3 visualization run: viz_rows=%d pair_count=%d elapsed=%s",
        len(viz_frame),
        len(pair_records),
        format_duration(metrics["elapsed_seconds"]),
    )
    return {
        "viz_frame": viz_frame,
        "pretrained_frame": pretrained_frame,
        "finetuned_frame": finetuned_frame,
        "finetune_pairs_path": finetune_pairs_path,
        "adapter_tar_path": adapter_tar_path,
        "metrics": metrics,
    }
