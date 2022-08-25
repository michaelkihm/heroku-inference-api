import pandas as pd


def test_cleaned_data_has_no_questionmark(cleaned_data: pd.DataFrame):
    assert '?' not in cleaned_data.values


def test_cleaned_data_has_no_null_valus(cleaned_data: pd.DataFrame):
    assert not cleaned_data.isnull().any().any()


def test_cleaning_removed_not_required_cols(cleaned_data):
    removed_cols = ["fnlgt", "capital-gain", "capital-loss", "education-num"]
    columns = cleaned_data.columns

    for removed_col in removed_cols:
        assert removed_col not in columns
