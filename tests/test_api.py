from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_return_200():
    res = client.get("/")

    assert res.status_code == 200
