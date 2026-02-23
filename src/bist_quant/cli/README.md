# `cli/` — Command-Line Tools

## Purpose

Administrative CLI scripts for cache management and data validation. These are not part of the library API — they are maintenance utilities.

## Files

```
cli/
├── cache_cli.py       # Cache inspection, clearing, and TTL management
└── validate_data.py   # Data file validation (schema checks, freshness, coverage)
```

Also note: `cli.py` exists at the `bist_quant/` root level as the top-level CLI entry point.

---

## `cache_cli.py`

Cache management tool for `DiskCache` contents:

```bash
python -m bist_quant.cli.cache_cli inspect          # show cache summary by category
python -m bist_quant.cli.cache_cli clear            # clear all cache
python -m bist_quant.cli.cache_cli clear --category prices  # clear one category
python -m bist_quant.cli.cache_cli expired          # list or remove expired entries
```

---

## `validate_data.py`

Validates all data files for schema correctness, staleness, and coverage:

```bash
python -m bist_quant.cli.validate_data              # validate all data files
python -m bist_quant.cli.validate_data --report     # save validation report to file
```

Checks performed:
- Prices file: required columns, date continuity, no future dates.
- Fundamentals: consolidated panel shape, `YYYY/MM` column format, coverage %.
- Regime labels: all values are valid `RegimeLabel` members.
- XAU/TRY and XU100 prices: continuity checks.

---

## Local Rules for Contributors

1. **CLI tools must handle `--help` gracefully.** All commands must provide argparse help strings.
2. **No side effects on import.** CLI entry points must be guarded by `if __name__ == "__main__":`.
3. **Exit codes:** Use `sys.exit(0)` for success, `sys.exit(1)` for errors — do not raise unhandled exceptions.
