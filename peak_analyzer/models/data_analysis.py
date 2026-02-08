"""
Data Analysis Models

Data structures for representing analysis results, data characteristics,
benchmarking results, and other metadata related to peak detection analysis.
"""

from dataclasses import dataclass
from typing import Any

# =============================================================================
# Data Characteristics and Analysis Results
# =============================================================================

@dataclass
class DataCharacteristics:
    """Characteristics of input data for strategy selection."""
    shape: tuple[int, ...]
    ndim: int
    data_type: str
    value_range: tuple[float, float]
    plateau_ratio: float
    noise_level: float
    peak_density_estimate: float


@dataclass
class BenchmarkResults:
    """Results from strategy benchmarking."""
    strategy_name: str
    execution_time: float
    memory_usage: float
    peaks_detected: int
    quality_metrics: dict[str, float]