
import os
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Shorts Maker Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Shorts Maker Backend is running"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    token = os.getenv("HF_TOKEN")

    if not token:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured"
        )

    audio = await file.read()

    if not audio:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty"
        )

    api_url = (
        "https://router.huggingface.co/"
        "hf-inference/models/openai/whisper-large-v3"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": file.content_type or "audio/mpeg"
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=audio,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Whisper API error: {str(error)}"
)
      
