# Portfolio Engine CLI Reference

## Standardized Command Format

The portfolio engine uses **positional arguments** for signal selection:

```bash
python portfolio_engine.py <signal_name>
```

## Available Signals

Run `python portfolio_engine.py --help` to see all available signals.

### Core Factor Signals
```bash
python portfolio_engine.py momentum
python portfolio_engine.py value
python portfolio_engine.py profitability
python portfolio_engine.py investment
python portfolio_engine.py size
```

### Technical Signals
```bash
python portfolio_engine.py sma
python portfolio_engine.py donchian
python portfolio_engine.py xu100
```

### Composite Signals
```bash
python portfolio_engine.py trend_value
python portfolio_engine.py breakout_value
```

### New Macro-Aware Signals
```bash
python portfolio_engine.py currency_rotation
python portfolio_engine.py dividend_rotation
python portfolio_engine.py macro_hedge
```

### Run All Signals
```bash
python portfolio_engine.py all
```

## Optional Arguments

### Custom Date Range
```bash
python portfolio_engine.py momentum --start-date 2020-01-01 --end-date 2024-12-31
```

### Legacy Format (Deprecated)
The old `--factor` flag still works for backward compatibility:
```bash
python portfolio_engine.py --factor momentum  # Still works, but not recommended
```

## Examples

### Run a single signal
```bash
cd /home/safa/Documents/Markets/BIST/Models
python portfolio_engine.py currency_rotation
```

### Run all signals
```bash
cd /home/safa/Documents/Markets/BIST/Models
python portfolio_engine.py all
```

### Run with custom date range
```bash
cd /home/safa/Documents/Markets/BIST/Models
python portfolio_engine.py dividend_rotation --start-date 2022-01-01 --end-date 2026-02-06
```

## Help
```bash
python portfolio_engine.py --help
```

## Output

Results are saved to:
```
Models/results/<signal_name>/
├── backtest_results.csv
├── yearly_metrics.csv
├── holdings_history.csv
└── performance_summary.txt
```

## Notes

- Signal names are automatically detected from `Models/configs/` directory
- Each signal must have a corresponding config file: `Models/configs/<signal_name>.py`
- Signals can be enabled/disabled in their config files
- The `--factor` flag is deprecated but still supported for backward compatibility
