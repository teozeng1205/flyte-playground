from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone

DEFAULT_INPUT_BUCKET = "s3-atp-3victors-3vprod-use1-derived-common-output"
DEFAULT_INPUT_PREFIX = "v1"
DEFAULT_OUTPUT_PREFIX = "s3://3v-teo-dev/dco_visualize/"
DEFAULT_RANDOM_SEED = 42
DEFAULT_TARGET_COLUMN = "price_inc"


def make_run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class DCOVisualizeConfig:
    customer: str = "AA"
    sales_date: str = field(default_factory=lambda: date.today().isoformat())
    sample_rows: int = 100_000
    train_rows: int = 50_000
    viz_rows: int = 500_000
    dashboard_point_cap: int = 200_000
    embedding_dims: int = 0
    output_prefix: str = DEFAULT_OUTPUT_PREFIX
    batch_size: int = 50_000
    random_seed: int = DEFAULT_RANDOM_SEED
    target_column: str = DEFAULT_TARGET_COLUMN
    tabpfn_version: str = "2.5"
    pretrained_n_estimators: int = 1
    pretrained_fit_mode: str = "fit_with_cache"
    pretrained_memory_saving_mode: bool | str = False
    finetune_epochs: int = 8
    finetune_learning_rate: float = 1e-5
    finetune_weight_decay: float = 0.01
    finetune_validation_split_ratio: float = 0.1
    finetune_ctx_plus_query_samples: int = 10_000
    finetune_ctx_query_split_ratio: float = 0.2
    finetune_inference_subsample_samples: int = 50_000
    finetune_early_stopping_patience: int = 4
    finetune_min_delta: float = 1e-4
    finetune_eval_winsor_quantile: float = 0.995
    finetune_n_estimators_finetune: int = 1
    finetune_n_estimators_validation: int = 1
    finetuned_n_estimators: int = 1
    finetune_inference_fit_mode: str = "fit_with_cache"
    finetune_inference_memory_saving_mode: bool | str = False
    n_preprocessing_jobs: int = 8
    embedding_n_fold: int = 0
    hdbscan_min_cluster_size: int = 128
    hdbscan_min_samples: int = 16
    umap_neighbors: int = 30
    umap_min_dist: float = 0.05
    progress_log_every_batches: int = 5
    progress_log_every_files: int = 10
    max_hover_columns: int = 12
    max_top_routes: int = 20
    max_matrix_origins: int = 14
    max_matrix_destinations: int = 14
    max_fingerprint_values_per_feature: int = 8
    top_airport_market_coverage_n: int = 1_000
    top_source_coverage_n: int = 20
    top_carrier_coverage_n: int = 20
    route_source_column: str = "origin_metro"
    route_destination_column: str = "destination_metro"
    departure_date_column: str = "outbound_departure_date"
    advance_purchase_column: str = "advance_purchase"
    return_date_column: str = "inbound_departure_date"

    def validate(self) -> None:
        if not self.customer:
            raise ValueError("customer must be non-empty")
        if not self.sales_date:
            raise ValueError("sales_date must be non-empty")
        if self.sample_rows <= 0:
            raise ValueError("sample_rows must be positive")
        if self.train_rows <= 0:
            raise ValueError("train_rows must be positive")
        if self.viz_rows <= 0:
            raise ValueError("viz_rows must be positive")
        if self.dashboard_point_cap <= 0:
            raise ValueError("dashboard_point_cap must be positive")
        if self.train_rows > self.viz_rows:
            raise ValueError("train_rows cannot exceed viz_rows")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.embedding_n_fold < 0:
            raise ValueError("embedding_n_fold must be non-negative")
        if self.embedding_n_fold == 1:
            raise ValueError("embedding_n_fold cannot be 1")
        if self.pretrained_n_estimators <= 0:
            raise ValueError("pretrained_n_estimators must be positive")
        if self.pretrained_fit_mode not in {"low_memory", "fit_preprocessors", "fit_with_cache", "batched"}:
            raise ValueError("pretrained_fit_mode must be a valid TabPFN fit mode")
        if self.finetune_epochs <= 0:
            raise ValueError("finetune_epochs must be positive")
        if self.finetune_n_estimators_finetune <= 0:
            raise ValueError("finetune_n_estimators_finetune must be positive")
        if self.finetune_n_estimators_validation <= 0:
            raise ValueError("finetune_n_estimators_validation must be positive")
        if self.finetuned_n_estimators <= 0:
            raise ValueError("finetuned_n_estimators must be positive")
        if self.finetune_inference_fit_mode not in {"low_memory", "fit_preprocessors", "fit_with_cache", "batched"}:
            raise ValueError("finetune_inference_fit_mode must be a valid TabPFN fit mode")
        if not 0.5 < self.finetune_eval_winsor_quantile <= 1.0:
            raise ValueError("finetune_eval_winsor_quantile must be in (0.5, 1.0]")
        if self.n_preprocessing_jobs <= 0:
            raise ValueError("n_preprocessing_jobs must be positive")
        if self.hdbscan_min_cluster_size <= 1:
            raise ValueError("hdbscan_min_cluster_size must be greater than 1")
        if self.hdbscan_min_samples <= 0:
            raise ValueError("hdbscan_min_samples must be positive")
        if self.umap_neighbors <= 1:
            raise ValueError("umap_neighbors must be greater than 1")
        if self.progress_log_every_batches <= 0:
            raise ValueError("progress_log_every_batches must be positive")
        if self.progress_log_every_files <= 0:
            raise ValueError("progress_log_every_files must be positive")
        if self.top_airport_market_coverage_n <= 0:
            raise ValueError("top_airport_market_coverage_n must be positive")
        if self.top_source_coverage_n <= 0:
            raise ValueError("top_source_coverage_n must be positive")
        if self.top_carrier_coverage_n <= 0:
            raise ValueError("top_carrier_coverage_n must be positive")
        if not self.target_column:
            raise ValueError("target_column must be non-empty")
        if not self.output_prefix.startswith("s3://"):
            raise ValueError("output_prefix must be an s3:// URI")

    @property
    def input_uri(self) -> str:
        year, month, day = self.sales_date.split("-")
        return f"s3://{DEFAULT_INPUT_BUCKET}/{DEFAULT_INPUT_PREFIX}/{self.customer}/{year}/{month}/{day}/"

    def run_output_prefix(self, run_timestamp: str) -> str:
        base = self.output_prefix.rstrip("/")
        return (
            f"{base}/tabpfn_v25/customer={self.customer}/sales_date={self.sales_date}/"
            f"run_ts={run_timestamp}/"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
