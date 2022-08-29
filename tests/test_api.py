from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root():
    res = client.get("/")

    assert res.status_code == 200
    assert res.json()["message"] == "RandomForest inference API for census dataset"


def test_inference_endpoint():
    request_body = {
        "age": 23,
        "workclass": "Private",
        "education": "11th",
        "marital-status": "Married-civ-spouse",
        "occupation": "Transport-moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    res = client.post("/inference", json=request_body)
    res_json = res.json()

    assert res.status_code == 200
    assert res_json["salary"] in ["<=50K", ">50K"]
