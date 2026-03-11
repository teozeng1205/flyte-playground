from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class ProgressSnapshot:
    done: int
    total: int
    percent: float
    elapsed_seconds: float
    rate_per_second: float
    remaining_seconds: float | None
    eta_utc: str | None


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def progress_snapshot(done: int, total: int, started_at: float) -> ProgressSnapshot:
    elapsed = max(0.0, time.perf_counter() - started_at)
    capped_done = min(max(done, 0), max(total, 0))
    percent = 0.0 if total <= 0 else (capped_done / total) * 100.0
    rate = 0.0 if elapsed <= 0 else capped_done / elapsed
    remaining = max(total - capped_done, 0)
    if rate <= 0:
        remaining_seconds = None
        eta_utc = None
    else:
        remaining_seconds = remaining / rate
        eta_utc = (
            datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
        ).strftime("%Y-%m-%d %H:%M:%SZ")
    return ProgressSnapshot(
        done=capped_done,
        total=total,
        percent=percent,
        elapsed_seconds=elapsed,
        rate_per_second=rate,
        remaining_seconds=remaining_seconds,
        eta_utc=eta_utc,
    )
