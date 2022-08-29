import pandas as pd
from mock import patch
from mock_open import MockOpen

from model_training.ml.data import CAT_FEATURES, process_data
from model_training.ml.model import data_slice_evaluation, inference


def test_inference_for_salary_lt_50k(models):
    data = pd.DataFrame(
        {
            "age": [43],
            "workclass": ["Private"],
            "education": ["11th"],
            "marital-status": ["Married-civ-spouse"],
            "occupation": ["Transport-moving"],
            "relationship": ["Husband"],
            "race": ["White"],
            "sex": ["Male"],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )
    model, encoder, lb = models
    X, *_ = process_data(data, CAT_FEATURES, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    assert len(y_pred) == 1
    assert y_pred[0] == 0, "Predicted salary should be 0 (<=50K)"


def test_inference_for_salary_gt_50k(models):
    data = pd.DataFrame(
        {
            "age": [40],
            "workclass": ["Private"],
            "education": ["Masters"],
            "marital-status": ["Married-civ-spouse"],
            "occupation": ["Exec-managerial"],
            "relationship": ["Husband"],
            "race": ["White"],
            "sex": ["Male"],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )
    model, encoder, lb = models
    X, *_ = process_data(data, CAT_FEATURES, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    assert len(y_pred) == 1
    assert y_pred[0] == 1, "Predicted salary should be 1 (>50K)"


def test_data_slice_evaluation(cleaned_data, models):
    expected_calls = (
        len([value for col in CAT_FEATURES for value in cleaned_data[col].unique()])
    ) * 2
    model, encoder, lb = models
    with patch("builtins.open", MockOpen()) as f_mock:
        data_slice_evaluation(cleaned_data, model, encoder, lb, CAT_FEATURES)

        assert len(f_mock.return_value.write.mock_calls) == expected_calls
