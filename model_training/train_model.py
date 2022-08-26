# Script to train machine learning model.

import logging

import pandas as pd
from ml.data import CAT_FEATURES, clean_data, process_data
from ml.model import evaluate_model, save_models, train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename="./logs/training.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


logging.info("Load and process train data")
data = pd.read_csv("./data/census.csv", skipinitialspace=True)

data = clean_data(data)
data.to_csv("./data/cleaned_census.csv", index=False)

logging.info("Split data")
train, test = train_test_split(data, test_size=0.20)


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label="salary", training=True
)


logging.info("Start model training")
model = train_model(X_train, y_train)
save_models(model, encoder, lb)

# Evaluate model
precision, recall, fbeta = evaluate_model(test, CAT_FEATURES)
logging.info(f"Evaluated model with following scores: \n- precision: {precision}\n- recall: {recall}\n- fbeta: {fbeta}")
