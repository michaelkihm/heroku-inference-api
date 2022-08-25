
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data

MODEL_DIR = "models"


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(max_depth=10, random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)  # type: ignore
    precision = precision_score(y, preds, zero_division=1)  # type: ignore
    recall = recall_score(y, preds, zero_division=1)  # type: ignore
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_models(model, encoder, label_binarizer):
    """ Saves Model, Enocder and LabelBinarizer
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(encoder, os.path.join(MODEL_DIR, 'encoder.pkl'))
    joblib.dump(label_binarizer, os.path.join(MODEL_DIR, 'label_binarizer.pkl'))


def load_models():
    model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
    lb = joblib.load(os.path.join(MODEL_DIR, 'label_binarizer.pkl'))

    return model, encoder, lb


def evaluate_model(test, cat_features):
    model, encoder, lb = load_models()

    X_test, y_test, *_ = process_data(test, cat_features, "salary", False, encoder, lb)

    y_pred = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    return precision, recall, fbeta
