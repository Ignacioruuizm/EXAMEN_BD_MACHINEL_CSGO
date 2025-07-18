import requests

BASE_URL = "http://localhost:8000"

def test_classification():
    test_data = {
        "RoundHeadshots": 2,
        "TeamStartingEquipmentValue": 4000,
        "PrimaryAssaultRifle": 1,
        "TimeAlive": 45.5
    }
    
    response = requests.post(f"{BASE_URL}/predict/clasificacion", json=test_data)
    print("Clasificación Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("\n")

def test_regression():
    test_data = {
        "RoundHeadshots": 1,
        "TeamStartingEquipmentValue": 3500,
        "PrimaryAssaultRifle": 0,
        "TimeAlive": 30.2
    }
    
    response = requests.post(f"{BASE_URL}/predict/regresion", json=test_data)
    print("Regresión Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("=== Probando Endpoints ===")
    test_classification()
    test_regression()