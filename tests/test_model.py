from campus_demand_forecasting.model import evaluate, generate_dataset, train_model


def test_dataset_is_deterministic_and_well_shaped():
    first = generate_dataset(30, seed=7)
    second = generate_dataset(30, seed=7)
    assert first.equals(second)
    assert first.shape == (30, 7)
    assert first["demand_units"].gt(0).all()


def test_model_beats_a_naive_mean_baseline():
    frame = generate_dataset(seed=7)
    result = train_model(frame, seed=7)
    naive_mae = abs(result.test_frame["actual"] - frame["demand_units"].mean()).mean()
    assert result.metrics["mae"] < naive_mae


def test_public_evaluate_contract():
    metrics = evaluate(seed=1)
    assert set(metrics) == {"mae", "rmse", "test_rows"}
    assert metrics["mae"] >= 0
