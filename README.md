# Campus Demand Forecasting

[![CI](https://github.com/ratnesh-ml/campus-demand-forecasting/actions/workflows/test.yml/badge.svg)](https://github.com/ratnesh-ml/campus-demand-forecasting/actions/workflows/test.yml)

I built this tabular ML baseline around a familiar planning question: if attendance, weekdays, weather, and exam periods move together, how could a campus team begin estimating daily resource demand? The point is not to claim that a synthetic dataset represents a real campus. The point is to practise the modelling habits I would want before making any real planning claim.

I create deterministic daily observations, define a data contract, compare a random-forest pipeline against a naive reference, and report error in units a planner can understand.

## At a glance

| Layer | What I implemented |
| --- | --- |
| Data | Deterministic synthetic daily campus observations. |
| Features | Calendar variables, exam period, rain, temperature, and attendance. |
| Model | Random forest regression in a scikit-learn pipeline. |
| Evaluation | MAE and RMSE, a held-out split, and a naive baseline. |
| Reproducibility | Fixed seeds, `pyproject.toml`, tests, and GitHub Actions. |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m campus_demand_forecasting
pytest -q
```

## The technical habit I was practising

I wanted to keep the data generation, training, and evaluation separate so the baseline was easier to inspect and change. The project includes a validated `time_ordered_split` helper for a leakage-aware chronological evaluation experiment. I keep that distinct from the default random holdout rather than implying the demo is already a production forecast.

## Limits and next experiments

The data is simulated and must not be used for operational decisions. The default baseline still uses a random holdout; the chronological helper is a safer starting point for the next experiment, not a substitute for real time-aware validation.

If I continue this project, I would use permission-cleared campus data, add prediction intervals, measure error across meaningful segments, and build a small planning interface around the model.

## Verification and license

Run `pytest -q` for local checks. GitHub Actions runs the regression suite on pushes and pull requests. MIT licensed; see [LICENSE](LICENSE).
