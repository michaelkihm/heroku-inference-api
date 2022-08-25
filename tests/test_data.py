import pandas as pd


def test_data_set_is_available():
    df = pd.read_csv("./data/census.csv")

    assert len(df.columns) > 1
