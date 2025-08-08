# utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import requests
from pathlib import Path
from typing import List

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")
genai.configure(api_key=API_KEY)

# --------------- Document extraction ---------------
def extract_text_from_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            p = page.extract_text()
            if p:
                text.append(p)
    return "\n".join(text)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def extract_text_from_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def fetch_url_text(url: str) -> str:
    # lightweight fetch; for production use a better reader (readability)
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return ""
    return r.text

def extract_all_text_from_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    if p.suffix.lower() in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    if p.suffix.lower() == ".txt":
        return extract_text_from_txt(path)
    # fallback: return raw bytes as string attempt
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except:
        return ""

# --------------- Gemini helpers ---------------
def upload_file_bytes_to_gemini_bytes(name: str, b: bytes, mime_type: str = None):
    """
    Upload bytes to Gemini and return the upload object that you can pass to generate_content.
    genai.upload_file can accept a path; to upload bytes we create a temp file.
    """
    # genai.upload_file requires a path. Write temp.
    tmp = Path("/tmp") / name
    tmp.write_bytes(b)
    # set mime_type when possible
    if mime_type:
        f = genai.upload_file(str(tmp), mime_type=mime_type)
    else:
        f = genai.upload_file(str(tmp))
    try:
        tmp.unlink()
    except:
        pass
    return f

def transcribe_audio_bytes_with_gemini(b: bytes, mime_type: str = "audio/wav"):
    """
    Upload audio bytes and ask Gemini to transcribe it. Returns plain text.
    We upload the file then reference it in the generate_content call.
    """
    upload = upload_file_bytes_to_gemini_bytes("segment.wav", b, mime_type=mime_type)
    model = genai.GenerativeModel("gemini-1.5-pro")
    # Construct a multimodal request where the file reference is provided; Gemini will transcribe.
    # Use a short prompt request to get a clean transcript.
    resp = model.generate_content([
        upload,
        {
            "text": "Transcribe the audio in the attached file. Return only the transcript text (no commentary)."
        }
    ])
    # resp.text contains transcript
    return resp.text

def analyze_text_with_gemini(prompt: str):
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    return resp.text
