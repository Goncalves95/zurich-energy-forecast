"""Tests for model training and prediction."""


def test_predict_returns_list():
    from models.predict import predict
    result = predict()
    assert isinstance(result, list)
