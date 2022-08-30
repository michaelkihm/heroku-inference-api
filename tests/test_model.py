import pandas as pd
from mock import patch
from mock_open import MockOpen
from sklearn.ensemble import RandomForestClassifier

from model_training.ml.data import CAT_FEATURES, process_data
from model_training.ml.model import (
    data_slice_evaluation,
    evaluate_model,
    inference,
    train_model,
)


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


def test_evaluates_the_model(cleaned_data, models):
    model, encoder, lb = models
    precision, recall, fbeta = evaluate_model(
        cleaned_data, CAT_FEATURES, model, encoder, lb
    )

    assert 0 <= precision <= 1, "precission should be a value between 0 and 1"
    assert 0 <= recall <= 1, "recall should be a value between 0 and 1"
    assert 0 <= fbeta <= 1, "fbeta should be a value between 0 and 1"


def test_trains_the_model(cleaned_data, models):
    # _, encoder, lb = models
    X_train, y_train, encoder, lb = process_data(
        cleaned_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X_train, y_train)

    assert isinstance(
        model, RandomForestClassifier
    ), "Model should be valid sklean RandomForestClassifier"
