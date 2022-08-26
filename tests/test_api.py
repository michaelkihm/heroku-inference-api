from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root():
    res = client.get("/")

    assert res.status_code == 200


def test_inference_endpoint():
    request_body = {
        "age": 23,
        "workclass": "Private",
        "education": "11th",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Transport-moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    }
    res = client.post("/inference", json=request_body)
    res_json = res.json()

    assert res.status_code == 200
    assert res_json["salary"] in ["<=50K", ">50K"]
