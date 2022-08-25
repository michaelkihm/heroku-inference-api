import pandas as pd


def test_cleaned_data_has_no_questionmark(cleaned_data: pd.DataFrame):
    assert '?' not in cleaned_data.values


def test_cleaned_data_has_no_null_valus(cleaned_data: pd.DataFrame):
    assert not cleaned_data.isnull().any().any()
