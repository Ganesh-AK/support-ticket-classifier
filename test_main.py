from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, Ticket Classifier"}


def test_predict_login():
    response = client.post("/predict", json={"text": "I forgot my password and my account is locked"})
    assert response.status_code == 200
    assert response.json()["category"] == "login"


def test_predict_payment():
    response = client.post("/predict", json={"text": "Money was debited but my refund is not received"})
    assert response.status_code == 200
    assert response.json()["category"] == "payment"


def test_predict_network():
    response = client.post("/predict", json={"text": "VPN is not connecting and internet is very slow"})
    assert response.status_code == 200
    assert response.json()["category"] == "network"


def test_predict_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422