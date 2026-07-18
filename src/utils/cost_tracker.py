"""
Cost tracker — estimates per-document inference cost.

Model used: CPU-based PaddleOCR + YOLOv8n. Pricing is amortized from typical
cloud-CPU rates (e.g., AWS t3.large ~ $0.083/hr or equivalents).

Cost(doc) = runtime_sec * ($/sec on target instance)
"""
from __future__ import annotations

# AWS t3.large on-demand ~ $0.0832/hr = $2.31e-5 per sec
# We use a slightly higher figure to account for amortized overheads.
CPU_USD_PER_SEC = 3.0e-5


def estimate_cost_usd(runtime_sec: float, rate_per_sec: float = CPU_USD_PER_SEC) -> float:
    """Linear cost estimate given wall-clock runtime."""
    return round(runtime_sec * rate_per_sec, 6)
