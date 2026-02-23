# Contributing

See the full guidelines in [CONTRIBUTING.md](https://github.com/Safa675/bist-quant/blob/main/CONTRIBUTING.md).

## Quick Reference

### Code Conventions

- `from __future__ import annotations` at the top of every file
- `frozen=True` for all config and path dataclasses
- No bare `except:` — always catch a specific exception type
- Atomic writes: write to `.tmp` then `os.replace()` (never partial writes)
- All secret comparisons use `hmac.compare_digest` (never `==`)
- 252-day annualization everywhere (not 250 or 365)

### Data Rules

- **borsapy is the only data source inside the library core.** yfinance is only allowed in `fetchers/` scripts.
- Never use `yfinance` in `clients/`, `signals/`, `engines/`, or `services/`.
- Period columns use `"YYYY/MM"` string format (not `pd.Period`).
- DatetimeIndex must be timezone-naive and floored to midnight before any join.
- `UFRS_TICKERS` in `clients/borsapy_client.py` governs which tickers use the alternate isyatirim endpoint.

### Signal Rules

- Every signal must call `validate_signal_panel_schema()` before returning.
- Output shape: `(dates × tickers)`, dtype `float64`, higher = more attractive.
- `NaN` means exclude — never substitute 0.0 for a missing signal.
- `signal_lag_days=1` is mandatory (anti-lookahead). Never skip.
- Turkish fundamental reporting lags: **Q4 = 70 days**, **Q1-Q3 = 40 days**.

### Analytics Rules

- `core_metrics.py` must remain dependency-free (no numpy, no pandas).
- The `list[SeriesPoint]` interface is the boundary — do not accept DataFrames.
- PRNG in analytics uses `_Xorshift32` — do not switch to `random.random()` or numpy.

### Regime Classifier

- MA window = **50** (tuned for BIST). Do not change to 200.
- Regime labels: `Bull`, `Bear`, `Recovery`, `Stress`. Do not add new labels without updating all downstream consumers.

### Testing

```bash
pytest tests/
pytest tests/unit/
pytest tests/integration/
```

### Pre-Commit Checklist

1. `from __future__ import annotations` present
2. BIST data source is borsapy (not yfinance) in library core
3. `validate_signal_panel_schema()` called before signal return
4. `signal_lag_days=1` applied
5. Atomic writes used for any file output
6. No hardcoded paths (`DataPaths` for all paths)
7. `frozen=True` on new dataclasses
8. Tests pass
