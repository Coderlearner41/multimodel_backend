# diarizer.py
import tempfile
import os
import numpy as np
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

encoder = VoiceEncoder()

def ensure_wav(mpath: str) -> str:
    """
    Ensure the audio is a mono 16k WAV. Returns path to wav file (may be new).
    """
    base, ext = os.path.splitext(mpath)
    out = base + "_16k.wav"
    if os.path.exists(out):
        return out
    audio = AudioSegment.from_file(mpath)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out, format="wav")
    return out

def diarize_file(path: str, win_s: float = 1.5, hop_s: float = 0.75, max_speakers: int = 2):
    """
    Return list of segments: [{"speaker":0/1, "start":float, "end":float}, ...]
    Using Resemblyzer embeddings + AgglomerativeClustering.
    """
    wav_path = ensure_wav(path)
    wav = preprocess_wav(wav_path)  # numpy array
    sr = 16000
    win_len = int(win_s * sr)
    hop_len = int(hop_s * sr)
    embeddings = []
    times = []

    if len(wav) < win_len:
        # pad with zeros if too short
        pad = np.zeros(win_len - len(wav))
        wav = np.concatenate([wav, pad])

    for start in range(0, len(wav) - win_len + 1, hop_len):
        seg = wav[start:start + win_len]
        emb = encoder.embed_utterance(seg)
        embeddings.append(emb)
        times.append((start / sr, (start + win_len) / sr))

    if len(embeddings) == 0:
        return []

    X = np.vstack(embeddings)
    # cluster into up to max_speakers (if samples < n_clusters, fallback)
    n_clusters = min(max_speakers, X.shape[0])
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    labels = clustering.labels_

    # merge adjacent windows with same label into larger segments
    segments = []
    cur_label = labels[0]
    cur_start = times[0][0]
    cur_end = times[0][1]
    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_end = times[i][1]
        else:
            segments.append({"speaker": int(cur_label), "start": float(cur_start), "end": float(cur_end)})
            cur_label = labels[i]
            cur_start, cur_end = times[i]
    segments.append({"speaker": int(cur_label), "start": float(cur_start), "end": float(cur_end)})

    # Sort by start time
    segments = sorted(segments, key=lambda s: s["start"])
    # Merge very close segments of same speaker
    merged = []
    for seg in segments:
        if merged and seg["speaker"] == merged[-1]["speaker"] and seg["start"] <= merged[-1]["end"] + 0.25:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    return merged
