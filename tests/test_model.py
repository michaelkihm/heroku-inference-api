import pandas as pd

from model_training.ml.data import CAT_FEATURES, process_data
from model_training.ml.model import inference


def test_inference(models):
    data = pd.DataFrame({
        "age": [20],
        "workclass": ["Private"],
        "education": ["Some-college"],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Exec-managerial"],
        "relationship": ["Husband"],
        "race": ["Black"],
        "sex": ["Male"],
        "hours-per-week": [40],
        "native-country": ["United-States"]
    })
    model, encoder, lb = models
    X, *_ = process_data(data, CAT_FEATURES, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    assert len(y_pred) == 1
