import os

import pandas as pd
from fastapi import FastAPI

from model_training.ml.data import CAT_FEATURES, process_data
from model_training.ml.model import inference, load_models

from .models import InferenceResponse, Person

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -r gdriveremote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "RandomForest inference API for census dataset"}


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(person: Person):
    data = pd.DataFrame(
        {
            "age": [person.age],
            "workclass": [person.workclass],
            "education": [person.education],
            "marital-status": [person.maritalStatus],
            "occupation": [person.occupation],
            "relationship": [person.relationship],
            "race": [person.race],
            "sex": [person.sex],
            "hours-per-week": [person.hoursPerWeek],
            "native-country": [person.nativeCountry],
        }
    )
    model, encoder, lb = load_models()
    X, *_ = process_data(data, CAT_FEATURES, training=False, encoder=encoder, lb=lb)
    y_pred = inference(model, X)

    return {"salary": ">50K" if y_pred[0] == 1 else "<=50K"}
