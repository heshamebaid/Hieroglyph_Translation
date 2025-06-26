import requests

url = "http://localhost:8080/chat"
data = {"query": "Who was Narmer?"}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    print("Response:", response.json())
except Exception as e:
    print("Error:", e) 