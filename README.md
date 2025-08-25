# Kubernetes_with_simple_chatbot

## Overview

This project is a simple chatbot API built with FastAPI and Hugging Face Transformers, designed for easy deployment in Kubernetes or Docker environments. The chatbot uses the `Qwen/Qwen2.5-0.5B-Instruct` model for generating responses and supports cross-origin requests via CORS middleware, making it easy to integrate with web frontends.

## Features

- FastAPI backend for high-performance async API
- Hugging Face Transformers for natural language generation
- CORS support for frontend-backend communication
- Easy deployment in Kubernetes or Docker

## API Endpoints

### POST /chat

Send a message to the chatbot and receive a generated response.

**Request:**
```json
{
  "message": "Hello, chatbot!"
}
```

**Response:**
```json
{
  "response": "Hi! How can I help you today?"
}
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Darshan0312/Kubernetes_with_simple_chatbot-.git
   cd Kubernetes_with_simple_chatbot-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn transformers
   ```

4. **Run the server:**
   ```bash
   python chatbot.py
   ```
   Or with Uvicorn:
   ```bash
   uvicorn chatbot:app --reload
   ```

## Kubernetes Deployment

To deploy on Kubernetes, create a Dockerfile and Kubernetes manifests (`deployment.yaml`, `service.yaml`). Example Dockerfile:

```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn transformers
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Customization

- **Model:** Change the model name in `chatbot.py` to use a different Hugging Face model.
- **CORS:** Restrict allowed origins in the CORS middleware for production security.


## Author

Darshan0312


