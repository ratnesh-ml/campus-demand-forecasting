# Campus Demand Forecasting

A small tabular machine-learning project about a problem I can imagine a campus actually facing: estimating daily resource demand when attendance, weather, weekdays, and exam periods move together.

The point is not to claim that a synthetic dataset represents a real campus. The point is to show a complete modelling habit: define a data contract, make a repeatable baseline, compare against a naive reference, and report error in units a planner can understand.

## What it demonstrates

| Layer | Choice |
| --- | --- |
| Data | Deterministic synthetic daily campus observations |
| Features | Calendar, exam period, rain, temperature, attendance |
| Model | Random forest regression in a scikit-learn pipeline |
| Evaluation | MAE and RMSE with a held-out split plus naive baseline |
| Reproducibility | Fixed seeds, `pyproject.toml`, tests, GitHub Actions |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m campus_demand_forecasting
pytest -q
```

## Why this is useful in a portfolio

It is deliberately more than a notebook. The code separates data generation, training, and evaluation, while the tests protect the public behaviour. A next iteration would use real, permission-cleared campus data, time-based validation, prediction intervals, and a small planning dashboard.

## Limitations

The data is simulated and should not be used for operational decisions. The random split is acceptable for this learning example but a production forecast would use chronological validation and monitor changes in the data-generating process.

## License

MIT. See [LICENSE](LICENSE).
