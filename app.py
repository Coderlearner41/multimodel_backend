# app.py
import os
import io
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from diarizer import diarize_file
from utils import extract_all_text_from_path, transcribe_audio_bytes_with_gemini, analyze_text_with_gemini, upload_file_bytes_to_gemini_bytes, fetch_url_text

load_dotenv()
DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "documents"
AUDIO_DIR = DATA_DIR / "audio"

app = FastAPI(title="Playground Backend (Multimodal)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Mock auth ----------------
@app.post("/api/login")
async def login(email: str = Form(...)):
    # In a real app: validate, create session/JWT. Here: return a mock token
    token = {"token": f"demo-token-{email}", "email": email}
    return JSONResponse(token)

# ---------------- Conversation Analysis ----------------
@app.post("/api/conversation")
async def conversation(file: UploadFile = File(...)):
    """
    Accepts an mp3 (or other) audio file.
    Steps:
    1. Save uploaded file to /tmp
    2. Run local diarization to get segments with start/end (seconds)
    3. For each diarized segment, extract the bytes corresponding to that time window (using pydub)
    4. Send each segment's bytes to Gemini for transcription (we do not use a STT vendor for diarization)
    5. Return combined transcript and speaker mapping.
    """
    # 1) save upload to temp
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_path = Path("/tmp") / f"upload{ext}"
    contents = await file.read()
    temp_path.write_bytes(contents)

    # 2) run diarizer to get segments (speaker, start, end)
    segments = diarize_file(str(temp_path), max_speakers=2)

    # 3) load with pydub to slice bytes per segment
    from pydub import AudioSegment
    audio = AudioSegment.from_file(temp_path)
    transcripts_by_segment = []
    full_transcript_parts = []

    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        slice_audio = audio[start_ms:end_ms]
        # export slice to bytes (wav)
        buf = io.BytesIO()
        slice_audio = slice_audio.set_frame_rate(16000).set_channels(1)
        slice_audio.export(buf, format="wav")
        bts = buf.getvalue()

        # 4) transcribe segment with Gemini
        try:
            text = transcribe_audio_bytes_with_gemini(bts, mime_type="audio/wav")
        except Exception as e:
            text = f"[TRANSCRIPTION_ERROR: {str(e)}]"

        transcripts_by_segment.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"], "text": text})
        full_transcript_parts.append(text)

    full_transcript = "\n".join(full_transcript_parts).strip()

    # Basic mapping to speakers (merge contiguous same speaker)
    # We'll provide an ordered list of blocks: speaker, start, end, text
    response = {
        "transcript": full_transcript,
        "segments": transcripts_by_segment,
        "raw_diarization": segments
    }

    return JSONResponse(response)


# ---------------- Image Analysis ----------------
@app.post("/api/image")
async def analyze_image(file: UploadFile = File(...)):
    # Save file temporarily
    ext = Path(file.filename).suffix.lower()
    allowed = [".png", ".jpg", ".jpeg", ".webp"]
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    tmp = Path("/tmp") / file.filename
    content = await file.read()
    tmp.write_bytes(content)

    # Upload bytes to Gemini and ask for a detailed description
    upload = upload_file_bytes_to_gemini_bytes(file.filename, content, mime_type="image/jpeg")
    prompt = f"""
    The attached image should be analyzed in detail. Provide:
    1) A 2-3 sentence descriptive caption.
    2) 6 bullet points of notable details (objects, colors, positions).
    3) Any inferred context or likely scenario.
    Return JSON with keys: caption, bullets (list), inference.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        resp = model.generate_content([upload, {"text": prompt}])
        text = resp.text
    except Exception as e:
        # Fallback: call analyze_text_with_gemini on basic prompt using labels
        text = f"ERROR in Gemini image analysis: {e}"

    # Return whatever Gemini produced
    return JSONResponse({"description": text})


# ---------------- Document / URL Summarization ----------------
@app.post("/api/summarize")
async def summarize(file: UploadFile = File(None), url: str = Form(None)):
    """
    Accept either a file upload (pdf/docx/txt) OR a url (form field 'url').
    Extract text, then ask Gemini to summarize.
    """
    text = ""
    if file is not None:
        tmp = Path("/tmp") / file.filename
        content = await file.read()
        tmp.write_bytes(content)
        text = extract_all_text_from_path(str(tmp))
    elif url:
        text = fetch_url_text(url)
    else:
        raise HTTPException(status_code=400, detail="Provide a file or url")

    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Could not extract text from input")

    # Trim large text to a reasonable chunk (Gemini can accept large contexts but keep prompt concise)
    preview = text[:20000]  # first N chars as context; adjust if you need more
    prompt = f"""
    Summarize the following content into:
    - A concise summary (max 150 words)
    - 5 bullet takeaways
    - If any action items are obvious, list them (max 5)
    
    Content:
    {preview}
    """

    from utils import analyze_text_with_gemini
    summary = analyze_text_with_gemini(prompt)
    return JSONResponse({"summary": summary})


# ---------------- Simple health ----------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}
