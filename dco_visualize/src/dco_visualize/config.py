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
class DCOVisualizeConfig:
    customer: str = "AA"
    sales_date: str = field(default_factory=lambda: date.today().isoformat())
    sample_rows: int = 100_000
    train_rows: int = 1_000_000
    viz_rows: int = 200_000
    embedding_dims: int = 128
    output_prefix: str = DEFAULT_OUTPUT_PREFIX
    batch_size: int = 50_000
    train_batch_size: int = 4_096
    random_seed: int = DEFAULT_RANDOM_SEED
    encoder_backend: str = "ft_transformer_contrastive"
    segment_method: str = "hdbscan"
    transformer_width: int = 192
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    pretrain_epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    contrastive_temperature: float = 0.1
    corruption_rate: float = 0.15
    id_like_uniqueness_ratio: float = 0.98
    datetime_success_ratio: float = 0.90
    long_text_threshold: int = 64
    max_categories: int = 128
    max_hover_columns: int = 12
    min_segment_size: int = 64
    viz_candidate_multiplier: int = 4

    def validate(self) -> None:
        if not self.customer:
            raise ValueError("customer must be non-empty")
        if not self.sales_date:
            raise ValueError("sales_date must be non-empty")
        if self.sample_rows <= 0:
            raise ValueError("sample_rows must be positive")
        if self.viz_rows <= 0:
            raise ValueError("viz_rows must be positive")
        if self.train_rows <= 0:
            raise ValueError("train_rows must be positive")
        if self.embedding_dims < 2:
            raise ValueError("embedding_dims must be at least 2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.pretrain_epochs <= 0:
            raise ValueError("pretrain_epochs must be positive")
        if self.transformer_width <= 0 or self.transformer_heads <= 0 or self.transformer_layers <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.transformer_width % self.transformer_heads != 0:
            raise ValueError("transformer_width must be divisible by transformer_heads")
        if not self.output_prefix.startswith("s3://"):
            raise ValueError("output_prefix must be an s3:// URI")

    @property
    def input_uri(self) -> str:
        year, month, day = self.sales_date.split("-")
        return f"s3://{DEFAULT_INPUT_BUCKET}/{DEFAULT_INPUT_PREFIX}/{self.customer}/{year}/{month}/{day}/"

    def run_output_prefix(self, run_timestamp: str) -> str:
        base = self.output_prefix.rstrip("/")
        return (
            f"{base}/v1/customer={self.customer}/sales_date={self.sales_date}/"
            f"run_ts={run_timestamp}/"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
