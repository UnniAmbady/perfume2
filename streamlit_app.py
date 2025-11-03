# import streamlit as st

import os
import requests
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- HeyGen API Configuration ---
BASE = "https://api.heygen.com/v1"
HEADERS_XAPI = {
    "accept": "application/json",
    "x-api-key": HEYGEN_API_KEY,
    "Content-Type": "application/json",
}

def _headers_bearer(tok: str):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }

# --- FastAPI App ---
app = FastAPI(title="Perfume App Backend")

# Add CORS middleware to allow your frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/api/start-session")
async def start_session():
    """
    Called once by the frontend on load.
    Creates a new HeyGen session and returns all necessary tokens.
    """
    try:
        # 1. Create a new session
        avatar_id = "June_HR_public"  # Fixed from your old script
        voice_id = "68dedac41a9f46a6a4271a95c733823c"
        payload = {"avatar_id": avatar_id, "voice_id": voice_id}
        
        r_new = requests.post(f"{BASE}/streaming.new", headers=HEADERS_XAPI, data=json.dumps(payload), timeout=60)
        r_new.raise_for_status()
        data = r_new.json().get("data", {})
        
        session_id = data.get("session_id")
        offer_sdp = (data.get("offer") or {}).get("sdp")
        
        # Determine RTC config
        ice = data.get("ice_servers2") or data.get("ice_servers") or [{"urls": ["stun:stun.l.google.com:19302"]}]
        rtc_config = {"iceServers": ice}

        if not session_id or not offer_sdp:
            raise HTTPException(status_code=500, detail="HeyGen /streaming.new failed to return session_id or offer.")

        # 2. Create a session token
        r_token = requests.post(f"{BASE}/streaming.create_token", headers=HEADERS_XAPI, data=json.dumps({"session_id": session_id}), timeout=60)
        r_token.raise_for_status()
        token = (r_token.json().get("data") or {}).get("token")
        
        if not token:
            raise HTTPException(status_code=500, detail="HeyGen /streaming.create_token failed to return a token.")

        return {
            "session_id": session_id,
            "session_token": token,
            "offer_sdp": offer_sdp,
            "rtc_config": rtc_config,
            "avatar_name": "June HR" # Fixed from your old script
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/api/chat")
async def chat_with_openai(request: Request):
    """
    Called by the frontend to get a ChatGPT response.
    """
    try:
        body = await request.json()
        user_text = body.get("text")
        if not user_text:
            raise HTTPException(status_code=400, detail="No text provided.")
            
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clear, concise assistant."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        reply = completion.choices[0].message.content or ""
        return {"reply": reply}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_audio(request: Request):
    """
    Called by the frontend to transcribe user audio.
    It expects a file upload named 'audio'.
    """
    form = await request.form()
    audio_file = form.get("audio")

    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    try:
        # Note: The file object from form data needs a 'name' attribute for OpenAI client
        # We pass the file object directly to the client.
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.filename, audio_file.file, audio_file.content_type)
        )
        return {"text": transcription.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {str(e)}")

@app.post("/api/stop-session")
async def stop_session(request: Request):
    """
    Called by the frontend 'onbeforeunload' to neatly close the session.
    """
    try:
        body = await request.json()
        session_id = body.get("session_id")
        session_token = body.get("session_token")
        
        if not session_id or not session_token:
            return {"status": "ignored", "detail": "Missing session info."}
            
        requests.post(
            f"{BASE}/streaming.stop",
            headers=_headers_bearer(session_token),
            data=json.dumps({"session_id": session_id}),
            timeout=5
        )
        return {"status": "stopped"}
    except Exception:
        return {"status": "failed"}

# To run this app:
# 1. Install libraries: pip install -r requirements.txt
# 2. Run the server: uvicorn main:app --host 0.0.0.0 --port 8000
