# Campus Demand Forecasting

[![CI](https://github.com/ratnesh-ml/campus-demand-forecasting/actions/workflows/test.yml/badge.svg)](https://github.com/ratnesh-ml/campus-demand-forecasting/actions/workflows/test.yml)

> **Portfolio demo:** [Open the Ratnesh ML Lab showcase](https://ratnesh-ml-lab.vercel.app)

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

## Recent depth improvements

The project includes a validated `time_ordered_split` helper for leakage-aware chronological evaluation in addition to the original reproducible baseline. This keeps the random split demo explicit while giving the next experiment a safer forecasting protocol. GitHub Actions runs the regression suite on pushes and pull requests.

## Why this is useful in a portfolio

It is deliberately more than a notebook. The code separates data generation, training, and evaluation, while the tests protect the public behaviour. A next iteration would use real, permission-cleared campus data, prediction intervals, and a small planning dashboard.

## Limitations

The data is simulated and should not be used for operational decisions. The default baseline still uses a random holdout; the new chronological helper is available for a production-shaped experiment but does not itself create a real forecast or uncertainty interval.

## License

MIT. See [LICENSE](LICENSE).
