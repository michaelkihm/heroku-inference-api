import pandas as pd

from model_training.ml.data import CAT_FEATURES, process_data
from model_training.ml.model import inference


def test_inference(models):
    data = pd.DataFrame({
        "age": [43],
        "workclass": ["Private"],
        "education": ["11th"],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Transport-moving"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "hours-per-week": [40],
        "native-country": ["United-States"]
    })
    model, encoder, lb = models
    X, *_ = process_data(data, CAT_FEATURES, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    assert len(y_pred) == 1
    assert y_pred[0] == 1 or y_pred[0] == 0
