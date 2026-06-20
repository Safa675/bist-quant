"""Volume profile, Renko bricks, Point & Figure columns, and timeframe resampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .core_metrics import mean
from ._shared import (
    ChartTimeframe,
    PricePoint,
    _clamp,
    _group_key_by_timeframe,
    _to_fixed,
)


@dataclass
class VolumeProfileBucket:
    bucket: str
    volume: float
    share_pct: float
    is_poc: bool


@dataclass
class RenkoBrick:
    index: int
    date: str
    price: float
    direction: Literal["up", "down"]


@dataclass
class PointFigureColumn:
    column: int
    type: Literal["X", "O"]
    start_date: str
    end_date: str
    boxes: int
    start_price: float
    end_price: float


def build_volume_profile(
    points: list[PricePoint],
    bucket_count: int = 12,
) -> list[VolumeProfileBucket]:
    """Build a volume profile from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if not ordered:
        return []
    closes = [p.close for p in ordered]
    min_price = min(closes)
    max_price = max(closes)
    buckets = max(6, min(30, int(bucket_count)))
    width = max(1e-6, (max_price - min_price) / buckets)
    totals = [0.0] * buckets

    for i, pt in enumerate(ordered):
        prev_close = ordered[i - 1].close if i > 0 else pt.close
        move = abs(pt.close / prev_close - 1) if prev_close > 0 else 0
        synthetic_volume = pt.volume if pt.volume is not None else 100_000 + move * 1_600_000
        bi = int(_clamp(math.floor((pt.close - min_price) / width), 0, buckets - 1))
        totals[bi] += synthetic_volume

    total_vol = sum(totals)
    poc = max(totals) if totals else 0
    result: list[VolumeProfileBucket] = []
    for i, vol in enumerate(totals):
        lo = min_price + width * i
        hi = lo + width
        result.append(VolumeProfileBucket(
            bucket=f"{_to_fixed(lo, 2)} - {_to_fixed(hi, 2)}",
            volume=_to_fixed(vol, 0),
            share_pct=_to_fixed((vol / total_vol) * 100, 2) if total_vol > 0 else 0,
            is_poc=vol == poc and vol > 0,
        ))
    return result


def build_renko_bricks(
    points: list[PricePoint],
    brick_size_pct: float = 1.2,
) -> list[RenkoBrick]:
    """Build Renko bricks from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    avg_price = mean([p.close for p in ordered])
    brick_size = max(0.01, avg_price * (_clamp(brick_size_pct, 0.1, 10) / 100))
    bricks: list[RenkoBrick] = []
    anchor = ordered[0].close
    guard = 0

    for i in range(1, len(ordered)):
        price = ordered[i].close
        diff = price - anchor
        while diff >= brick_size:
            anchor += brick_size
            bricks.append(RenkoBrick(index=len(bricks) + 1, date=ordered[i].date,
                                     price=_to_fixed(anchor, 3), direction="up"))
            diff = price - anchor
            guard += 1
            if guard > 50_000:
                break
        while diff <= -brick_size:
            anchor -= brick_size
            bricks.append(RenkoBrick(index=len(bricks) + 1, date=ordered[i].date,
                                     price=_to_fixed(anchor, 3), direction="down"))
            diff = price - anchor
            guard += 1
            if guard > 50_000:
                break
        if guard > 50_000:
            break
    return bricks[-320:]


def build_point_figure_columns(
    points: list[PricePoint],
    box_size_pct: float = 1.0,
    reversal: int = 3,
) -> list[PointFigureColumn]:
    """Build Point & Figure chart columns from price data."""
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    avg_price = mean([p.close for p in ordered])
    box_size = max(0.01, avg_price * (_clamp(box_size_pct, 0.2, 8) / 100))
    reversal_boxes = max(2, min(6, int(reversal)))
    columns: list[PointFigureColumn] = []

    current: dict[str, object] | None = None
    anchor = ordered[0].close

    for i in range(1, len(ordered)):
        price = ordered[i].close
        if current is None:
            if price >= anchor + box_size:
                boxes = max(1, int((price - anchor) / box_size))
                current = {"type": "X", "start_date": ordered[i - 1].date, "end_date": ordered[i].date,
                           "boxes": boxes, "start_price": anchor, "end_price": anchor + boxes * box_size}
                anchor = current["end_price"]
            elif price <= anchor - box_size:
                boxes = max(1, int((anchor - price) / box_size))
                current = {"type": "O", "start_date": ordered[i - 1].date, "end_date": ordered[i].date,
                           "boxes": boxes, "start_price": anchor, "end_price": anchor - boxes * box_size}
                anchor = current["end_price"]
            continue

        if current["type"] == "X":
            if price >= current["end_price"] + box_size:
                add = max(1, int((price - current["end_price"]) / box_size))
                current["boxes"] += add
                current["end_price"] += add * box_size
                current["end_date"] = ordered[i].date
                anchor = current["end_price"]
            elif price <= current["end_price"] - reversal_boxes * box_size:
                columns.append(PointFigureColumn(
                    column=len(columns) + 1, type=current["type"],
                    start_date=current["start_date"], end_date=current["end_date"],
                    boxes=current["boxes"],
                    start_price=_to_fixed(current["start_price"], 3),
                    end_price=_to_fixed(current["end_price"], 3),
                ))
                nb = max(reversal_boxes, int((current["end_price"] - price) / box_size))
                current = {"type": "O", "start_date": ordered[i].date, "end_date": ordered[i].date,
                           "boxes": nb, "start_price": current["end_price"],
                           "end_price": current["end_price"] - nb * box_size}
                anchor = current["end_price"]
        else:  # O column
            if price <= current["end_price"] - box_size:
                add = max(1, int((current["end_price"] - price) / box_size))
                current["boxes"] += add
                current["end_price"] -= add * box_size
                current["end_date"] = ordered[i].date
                anchor = current["end_price"]
            elif price >= current["end_price"] + reversal_boxes * box_size:
                columns.append(PointFigureColumn(
                    column=len(columns) + 1, type=current["type"],
                    start_date=current["start_date"], end_date=current["end_date"],
                    boxes=current["boxes"],
                    start_price=_to_fixed(current["start_price"], 3),
                    end_price=_to_fixed(current["end_price"], 3),
                ))
                nb = max(reversal_boxes, int((price - current["end_price"]) / box_size))
                current = {"type": "X", "start_date": ordered[i].date, "end_date": ordered[i].date,
                           "boxes": nb, "start_price": current["end_price"],
                           "end_price": current["end_price"] + nb * box_size}
                anchor = current["end_price"]

    if current:
        columns.append(PointFigureColumn(
            column=len(columns) + 1, type=current["type"],
            start_date=current["start_date"], end_date=current["end_date"],
            boxes=current["boxes"],
            start_price=_to_fixed(current["start_price"], 3),
            end_price=_to_fixed(current["end_price"], 3),
        ))
    return columns[-40:]


def build_synchronized_timeframes(
    points: list[PricePoint],
) -> dict[str, list[PricePoint]]:
    """Resample daily prices to weekly and monthly."""
    ordered = sorted(points, key=lambda p: p.date)
    output: dict[str, list[PricePoint]] = {"1D": ordered, "1W": [], "1M": []}

    for tf in ("1W", "1M"):
        grouped: dict[str, list[PricePoint]] = {}
        for row in ordered:
            key = _group_key_by_timeframe(row.date, tf)  # type: ignore[arg-type]
            grouped.setdefault(key, []).append(row)
        resampled: list[PricePoint] = []
        for _key in sorted(grouped):
            rows = grouped[_key]
            last = rows[-1]
            resampled.append(PricePoint(
                date=last.date,
                close=_to_fixed(last.close, 3),
                volume=_to_fixed(sum(r.volume or 0 for r in rows), 0),
            ))
        output[tf] = resampled
    return output


__all__ = [
    "VolumeProfileBucket",
    "RenkoBrick",
    "PointFigureColumn",
    "build_volume_profile",
    "build_renko_bricks",
    "build_point_figure_columns",
    "build_synchronized_timeframes",
]
