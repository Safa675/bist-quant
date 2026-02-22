# Multi-Asset Support

BIST Quant provides adapters for additional datasets beyond BIST equities.

## Optional Clients

- `CryptoClient`
- `USStockClient`
- `FXCommoditiesClient`
- `FundAnalyzer`
- `BorsapyAdapter`

These integrations may require optional dependency extras.

## Install Extras

```bash
pip install bist-quant[multi-asset]
pip install bist-quant[borsapy]
```

## Access via Top-Level API

```python
from bist_quant import CryptoClient, USStockClient

print(CryptoClient)
print(USStockClient)
```
