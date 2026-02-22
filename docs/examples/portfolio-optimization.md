# Portfolio Optimization Example

This example compares allocation styles (equal weight, inverse volatility, and mean-variance style workflows).

```python
import numpy as np
import pandas as pd

# returns: Date x Ticker matrix
returns = pd.DataFrame(...)

mean_returns = returns.mean() * 252
cov = returns.cov() * 252
weights = np.ones(len(mean_returns)) / len(mean_returns)
portfolio_return = float(mean_returns @ weights)
portfolio_vol = float(np.sqrt(weights @ cov.values @ weights))
```

Notebook version: [examples/03_portfolio_optimization.ipynb](https://github.com/Safa675/BIST/blob/main/examples/03_portfolio_optimization.ipynb)
