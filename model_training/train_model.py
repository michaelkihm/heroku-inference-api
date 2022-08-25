# Script to train machine learning model.

import logging

import pandas as pd
from ml.data import clean_data, process_data
from ml.model import evaluate_model, save_models, train_model
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

logging.basicConfig(
    filename="./logs/training.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


logging.info("Load and process train data")
# Add code to load in the data.
data = pd.read_csv("./data/cleaned_census.csv")

data = clean_data(data)
data.to_csv("./data/cleaned_census.csv")

logging.info("Split data")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


logging.info("Start model training")
model = train_model(X_train, y_train)
save_models(model, encoder, lb)

# Evaluate model
precision, recall, fbeta = evaluate_model(test, cat_features)
logging.info(f"Evaluated model with following scores: \n- precision: {precision}\n- recall: {recall}\n- fbeta: {fbeta}")
