# PharaohGuide Chatbot API

This is a FastAPI service for the PharaohGuide RAG chatbot.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```bash
   uvicorn chatbot_api:app --host 0.0.0.0 --port 8080 --reload
   ```

## Endpoints

- `POST /chat` — Chat with the RAG chatbot
  - Request body: `{ "query": "your question", "k": 5 }`
  - Response: `{ "answer": "..." }`

- `GET /health` — Health check
- `GET /config` — Get model/config info

## Example Usage

```bash
curl -X POST "http://localhost:8080/chat" -H "Content-Type: application/json" -d '{"query": "Who was Narmer?"}'
```

## Testing

### 1. Using curl
```bash
curl -X POST "http://localhost:8080/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who was Narmer?"}'
```

### 2. Using Python
Create a file `test_chatbot_api.py` with:
```python
import requests
url = "http://localhost:8080/chat"
data = {"query": "Who was Narmer?"}
response = requests.post(url, json=data)
print(response.json())
```
Run it with:
```bash
python test_chatbot_api.py
```

### 3. Using Swagger UI
Open your browser at:
```
http://localhost:8080/docs
```
You can interactively test all endpoints there.

### 4. Health Check
```bash
curl http://localhost:8080/health
```

### 5. Config Check
```bash
curl http://localhost:8080/config
```

## Notes
- Make sure your Hugging Face API token is set if required by your models.
- The service loads context documents from `data.py`. 