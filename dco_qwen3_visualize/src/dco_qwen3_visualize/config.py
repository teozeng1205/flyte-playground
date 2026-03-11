from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone

DEFAULT_INPUT_BUCKET = "s3-atp-3victors-3vprod-use1-derived-common-output"
DEFAULT_INPUT_PREFIX = "v1"
DEFAULT_OUTPUT_PREFIX = "s3://3v-teo-dev/dco_visualize/"
DEFAULT_RANDOM_SEED = 42


def make_run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class DCOQwen3VisualizeConfig:
    customer: str = "AA"
    sales_date: str = field(default_factory=lambda: date.today().isoformat())
    sample_rows: int = 100_000
    train_rows: int = 50_000
    viz_rows: int = 50_000
    output_prefix: str = DEFAULT_OUTPUT_PREFIX
    batch_size: int = 50_000
    random_seed: int = DEFAULT_RANDOM_SEED
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    target_column: str = "price_inc"
    row_instruction: str = (
        "Represent an airfare offer row so that rows with similar market, schedule, "
        "and fare behavior are close in embedding space."
    )
    sequence_max_length: int = 512
    embedding_dim: int = 512
    inference_batch_size: int = 32
    finetune_epochs: int = 3
    finetune_learning_rate: float = 5e-5
    finetune_weight_decay: float = 0.01
    finetune_batch_size: int = 16
    finetune_gradient_accumulation_steps: int = 4
    finetune_temperature: float = 0.05
    finetune_positive_price_tolerance: float = 0.15
    finetune_positive_day_tolerance: int = 14
    finetune_max_negatives: int = 4
    finetune_pair_rows: int = 20_000
    finetune_max_steps_per_epoch: int = 2_000
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    segment_clusters: int = 8
    layout_fit_rows: int = 10_000
    trustworthiness_rows: int = 5_000
    dashboard_point_cap: int = 50_000
    progress_log_every_batches: int = 5
    top_airport_market_coverage_n: int = 1_000
    top_source_coverage_n: int = 20
    top_carrier_coverage_n: int = 20

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
        if self.train_rows > self.viz_rows:
            raise ValueError("train_rows cannot exceed viz_rows")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.sequence_max_length <= 0:
            raise ValueError("sequence_max_length must be positive")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        if self.finetune_epochs <= 0:
            raise ValueError("finetune_epochs must be positive")
        if self.finetune_batch_size <= 0:
            raise ValueError("finetune_batch_size must be positive")
        if self.finetune_gradient_accumulation_steps <= 0:
            raise ValueError("finetune_gradient_accumulation_steps must be positive")
        if self.finetune_temperature <= 0:
            raise ValueError("finetune_temperature must be positive")
        if not 0 < self.finetune_positive_price_tolerance <= 1:
            raise ValueError("finetune_positive_price_tolerance must be in (0, 1]")
        if self.finetune_positive_day_tolerance < 0:
            raise ValueError("finetune_positive_day_tolerance must be non-negative")
        if self.finetune_max_negatives <= 0:
            raise ValueError("finetune_max_negatives must be positive")
        if self.finetune_pair_rows <= 0:
            raise ValueError("finetune_pair_rows must be positive")
        if self.finetune_max_steps_per_epoch <= 0:
            raise ValueError("finetune_max_steps_per_epoch must be positive")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.segment_clusters < 2:
            raise ValueError("segment_clusters must be at least 2")
        if self.layout_fit_rows <= 0:
            raise ValueError("layout_fit_rows must be positive")
        if self.trustworthiness_rows <= 0:
            raise ValueError("trustworthiness_rows must be positive")
        if self.dashboard_point_cap <= 0:
            raise ValueError("dashboard_point_cap must be positive")
        if self.progress_log_every_batches <= 0:
            raise ValueError("progress_log_every_batches must be positive")
        if not self.output_prefix.startswith("s3://"):
            raise ValueError("output_prefix must be an s3:// URI")

    @property
    def input_uri(self) -> str:
        year, month, day = self.sales_date.split("-")
        return f"s3://{DEFAULT_INPUT_BUCKET}/{DEFAULT_INPUT_PREFIX}/{self.customer}/{year}/{month}/{day}/"

    def run_output_prefix(self, run_timestamp: str) -> str:
        base = self.output_prefix.rstrip("/")
        return (
            f"{base}/qwen3_v1/customer={self.customer}/sales_date={self.sales_date}/"
            f"run_ts={run_timestamp}/"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
