import pandas as pd
import pytest

from model_training.ml.model import load_models


@pytest.fixture
def cleaned_data() -> pd.DataFrame:
    df = pd.read_csv("./data/cleaned_census.csv")
    return df


@pytest.fixture
def raw_data() -> pd.DataFrame:
    df = pd.read_csv("./data/census.csv")
    return df


@pytest.fixture
def models():
    model, encoder, lb = load_models()
    return model, encoder, lb
