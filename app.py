# app.py
from main import app  # Import your FastAPI app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)