"""Lightweight metrics collection with optional Prometheus integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - optional dependency
    Counter = None
    Gauge = None
    Histogram = None


def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


@dataclass
class MetricsCollector:
    """Collect in-memory metrics and mirror to Prometheus when available."""

    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(default_factory=dict)
    _prom_counters: dict[tuple[str, tuple[str, ...]], Any] = field(default_factory=dict)
    _prom_gauges: dict[tuple[str, tuple[str, ...]], Any] = field(default_factory=dict)
    _prom_histograms: dict[tuple[str, tuple[str, ...]], Any] = field(default_factory=dict)

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        label_items = _labels_key(labels)
        key = (name, label_items)
        self.counters[key] = float(self.counters.get(key, 0.0)) + float(value)
        counter = self._prom_counter(name, labels)
        if counter is not None:
            counter.inc(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        label_items = _labels_key(labels)
        key = (name, label_items)
        self.gauges[key] = float(value)
        gauge = self._prom_gauge(name, labels)
        if gauge is not None:
            gauge.set(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        label_items = _labels_key(labels)
        key = (name, label_items)
        bucket = self.histograms.setdefault(key, [])
        bucket.append(float(value))
        hist = self._prom_histogram(name, labels)
        if hist is not None:
            hist.observe(value)

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": {
                self._render_metric_key(name, labels): value
                for (name, labels), value in self.counters.items()
            },
            "gauges": {
                self._render_metric_key(name, labels): value
                for (name, labels), value in self.gauges.items()
            },
            "histograms": {
                self._render_metric_key(name, labels): list(values)
                for (name, labels), values in self.histograms.items()
            },
        }

    @staticmethod
    def _render_metric_key(name: str, labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return name
        rendered = ",".join(f"{k}={v}" for k, v in labels)
        return f"{name}{{{rendered}}}"

    def _prom_counter(self, name: str, labels: dict[str, str] | None) -> Any:
        if Counter is None:
            return None
        label_names = tuple(labels.keys()) if labels else ()
        key = (name, label_names)
        if key not in self._prom_counters:
            self._prom_counters[key] = Counter(name, f"{name} counter", list(label_names))
        metric = self._prom_counters[key]
        return metric.labels(**labels) if labels else metric

    def _prom_gauge(self, name: str, labels: dict[str, str] | None) -> Any:
        if Gauge is None:
            return None
        label_names = tuple(labels.keys()) if labels else ()
        key = (name, label_names)
        if key not in self._prom_gauges:
            self._prom_gauges[key] = Gauge(name, f"{name} gauge", list(label_names))
        metric = self._prom_gauges[key]
        return metric.labels(**labels) if labels else metric

    def _prom_histogram(self, name: str, labels: dict[str, str] | None) -> Any:
        if Histogram is None:
            return None
        label_names = tuple(labels.keys()) if labels else ()
        key = (name, label_names)
        if key not in self._prom_histograms:
            self._prom_histograms[key] = Histogram(name, f"{name} histogram", list(label_names))
        metric = self._prom_histograms[key]
        return metric.labels(**labels) if labels else metric
