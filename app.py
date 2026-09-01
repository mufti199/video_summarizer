"""
AV Digest — local Flask backend.

Pipeline:
  1. Download audio from a YouTube URL with yt-dlp.
  2. Transcribe the audio with faster-whisper (runs locally).
  3. Hold the transcript in a session, then summarize or chat about it on demand
     with a local open-weight model via Ollama.

Nothing leaves your machine. Requires: ffmpeg + Ollama running locally.
"""

import json
import os
import re
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import requests
from flask import Flask, Response, request, send_from_directory, stream_with_context

# --- Configuration ---------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # tiny/base/small/medium/large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 is fast on CPU

# Chapter breakdown: group the transcript into time windows and summarize each.
CHAPTER_SECONDS = int(os.environ.get("CHAPTER_SECONDS", "300"))  # window size (5 min)
CHAPTER_MIN_DURATION = int(os.environ.get("CHAPTER_MIN_DURATION", "360"))  # only chapter videos longer than this (6 min)

# Ollama defaults to a 4096-token window, which silently truncates a long
# transcript. Raise it, and keep what we send comfortably inside it.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
SUMMARY_CONTEXT_CHARS = int(os.environ.get("SUMMARY_CONTEXT_CHARS", "18000"))
CHAT_CONTEXT_CHARS = int(os.environ.get("CHAT_CONTEXT_CHARS", "12000"))
CHAT_HISTORY_TURNS = int(os.environ.get("CHAT_HISTORY_TURNS", "8"))  # user+assistant messages kept

app = Flask(__name__, static_folder=None)
HERE = Path(__file__).parent

# Transcripts live here between requests so the summarize and chat calls do not
# have to re-send them. In-memory and single-process on purpose: this is a
# local, single-user app, and a restart simply means transcribing again.
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))

# Lazily loaded so the server starts instantly and the model loads on first use.
_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
        )
    return _whisper


# --- Helpers ---------------------------------------------------------------

YT_RE = re.compile(
    r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)"
)


def is_youtube_url(url: str) -> bool:
    return bool(url) and bool(YT_RE.search(url))


def sse(event: str, **data) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def download_audio(url: str, dest_dir: str) -> tuple[str, str]:
    """Download bestaudio and return (audio_path, video_title)."""
    import yt_dlp

    out_tmpl = os.path.join(dest_dir, "audio.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube video")

    for f in os.listdir(dest_dir):
        if f.startswith("audio."):
            return os.path.join(dest_dir, f), title
    raise RuntimeError("Audio download failed — no output file produced.")


def transcribe(audio_path: str) -> list[dict]:
    """Return a list of {start, end, text} segments (timestamps in seconds)."""
    model = get_whisper()
    segments, _info = model.transcribe(audio_path, beam_size=1)
    out = []
    for seg in segments:
        txt = seg.text.strip()
        if txt:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": txt})
    return out


def full_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip()


def fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def make_chapters(segments: list[dict], window: int) -> list[dict]:
    """Group segments into fixed time windows. Returns [{start, end, text}]."""
    if not segments:
        return []
    chapters = []
    bucket, bucket_start = [], segments[0]["start"]
    for seg in segments:
        if seg["start"] - bucket_start >= window and bucket:
            chapters.append({
                "start": bucket_start,
                "end": bucket[-1]["end"],
                "text": " ".join(b["text"] for b in bucket),
            })
            bucket, bucket_start = [], seg["start"]
        bucket.append(seg)
    if bucket:
        chapters.append({
            "start": bucket_start,
            "end": bucket[-1]["end"],
            "text": " ".join(b["text"] for b in bucket),
        })
    return chapters


SUMMARY_PROMPT = """You are a precise assistant. Summarize the content below. Produce:

1. A one-sentence TL;DR.
2. 4-8 key points as a bulleted list.
3. A short "Takeaways" paragraph.

Write about the subject matter itself, never about the source. Start straight at item 1 \
with no preamble: no "This is a transcript", "This video", "In this video", "The speaker", \
"Here is a summary". Do not mention a transcript, video, audio or narrator anywhere. Refer \
to ideas and claims directly, not to who said them.

Be faithful to the content. Do not invent facts.

CONTENT:
\"\"\"
{transcript}
\"\"\"
"""


CHAPTER_PROMPT = """Below is one section of a longer piece of content. Reply with exactly \
this format:
TITLE: <a short 3-7 word title naming what is covered>
SUMMARY: <1-2 sentences giving the substance of it>

Name the specific topics and say what was actually claimed about each, rather than \
reporting that they came up. Write "Pod IPs are assigned by the CNI plugin, and IPAM is \
pluggable" rather than "This section focuses on pod IPs, CNI plugins and IPAM".

Do not add anything else. Do not open with "This section", "This part", "The speaker", \
"Here", or any other phrase pointing at the material rather than its subject. Do not \
mention a transcript, video, audio, section or speaker anywhere. Be faithful to the content.

SECTION:
\"\"\"
{transcript}
\"\"\"
"""


CHAT_SYSTEM = """You are answering questions about the content below, which the user has \
just transcribed.

Answer from the content. If it does not cover something, say so plainly instead of \
guessing, and say clearly when you are adding outside knowledge. Keep answers short \
unless asked for detail. Do not preface answers with "Based on the transcript" or similar; \
just answer.

TITLE: {title}

CONTENT:
\"\"\"
{transcript}
\"\"\"
"""


def new_session(title: str, segments: list[dict], transcript: str) -> str:
    """Store a transcript and return its session id, evicting the oldest if full."""
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "title": title,
        "segments": segments,
        "transcript": transcript,
        "messages": [],
    }
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return sid


def chat_stream(model: str, messages: list[dict]):
    """Yield assistant content chunks from Ollama's chat endpoint."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        obj = json.loads(line)
        piece = (obj.get("message") or {}).get("content")
        if piece:
            yield piece
        if obj.get("done"):
            break


def summarize_once(prompt: str, model: str) -> str:
    """Non-streaming Ollama call; returns the full response text."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# Models sometimes ignore the "no preamble" instruction, so strip it after the fact.
META_RE = re.compile(
    r"\btranscripts?\b"
    r"|^(sure|certainly|okay|ok|of course|absolutely)\b"
    r"|^here(?:'s| is)\b"
    r"|^below is\b",
    re.I,
)
# Framing lead-ins on a chapter line: "In this section, …", "This part covers …",
# "The speaker explains …". Stripping the lead leaves the topics themselves.
_SUBJECT = r"(?:this|the)\s+(?:section|part|segment|chapter|clip|video|episode|passage)"
_SPEAKER = r"(?:the\s+)?(?:speaker|narrator|author|presenter|host|instructor)"
_VERB = (
    r"(?:focus(?:es|ses)?\s+on|covers?|discusses|discuss|explains|explain|describes|"
    r"describe|examines|explores|introduces|outlines|highlights|reviews|presents|"
    r"looks?\s+at|deals?\s+with|talks?\s+about|goes?\s+over|walks?\s+through|"
    r"touches?\s+on|centers?\s+on|centres?\s+on|is\s+about)"
)
SECTION_LEAD_RE = re.compile(
    # A section subject only counts as framing if a comma or a framing verb follows it,
    # so a real sentence like "This part was recorded later" survives intact.
    rf"^(?:in\s+|during\s+)?(?:"
    rf"{_SUBJECT}(?:\s*[,:]\s*(?:{_SPEAKER}\s+{_VERB}\s+)?|\s+(?:{_VERB})\s+)"
    rf"|{_SPEAKER}\s+{_VERB}\s+"
    rf")(?:that\s+|how\s+|the\s+way\s+)?",
    re.I,
)


def _is_meta(sentence: str) -> bool:
    """True if a sentence frames the source rather than covering the content."""
    s = sentence.strip()
    if not s or s[0] in "-*#" or s[:2] in ("1.", "2.", "3."):
        return False
    return bool(META_RE.search(s))


def strip_preamble(text: str) -> str:
    """Drop leading lines/sentences that talk about the transcript instead of the topic."""
    lines = text.split("\n")
    while lines:
        if not lines[0].strip():
            lines.pop(0)
            continue
        kept = re.split(r"(?<=[.!?:])\s+", lines[0])
        while kept and _is_meta(kept[0]):
            kept.pop(0)
        if not kept:
            lines.pop(0)
            continue
        lines[0] = " ".join(kept)
        break
    return "\n".join(lines).lstrip()


def _recap(cleaned: str, original: str) -> str:
    """Re-capitalise only when a lead-in was actually removed, so that a line
    starting with something like "npm caching" keeps its own casing."""
    if cleaned and cleaned != original:
        return cleaned[0].upper() + cleaned[1:]
    return cleaned


def parse_chapter(text: str) -> tuple[str, str]:
    """Pull TITLE / SUMMARY out of a chapter model response, with fallbacks."""
    title, summary = "", ""
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif low.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    if not summary:
        summary = text.strip()
    cleaned = SECTION_LEAD_RE.sub("", strip_preamble(summary)).strip()
    if cleaned:  # keep the original rather than emptying a one-sentence summary
        summary = _recap(cleaned, summary)
    title = _recap(SECTION_LEAD_RE.sub("", title).strip(), title)
    if not title:
        title = (summary[:50] + "…") if len(summary) > 50 else summary
    return title, summary


def _ollama_stream(prompt: str, model: str):
    """Yield raw response chunks from Ollama as they arrive."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("response"):
            yield obj["response"]
        if obj.get("done"):
            break


def summarize_stream(transcript: str, model: str):
    """Stream the summary, holding the opening back long enough to strip any preamble."""
    prompt = SUMMARY_PROMPT.format(transcript=transcript[:SUMMARY_CONTEXT_CHARS])
    buf, cleaned = "", False
    for chunk in _ollama_stream(prompt, model):
        if cleaned:
            yield chunk
            continue
        buf += chunk
        # Wait for a complete first line (or a sentence-sized buffer) before deciding.
        if "\n" not in buf and len(buf) < 400:
            continue
        cleaned = True
        opening = strip_preamble(buf)
        if opening:
            yield opening
    if not cleaned:
        opening = strip_preamble(buf)
        if opening:
            yield opening


# --- Routes ----------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(HERE / "assets", filename)


@app.route("/models")
def models():
    """List locally installed Ollama models for the picker."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return {"models": names}
    except Exception as e:
        return {"models": [], "error": str(e)}, 200


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.route("/transcribe")
def transcribe_route():
    """Download and transcribe only. Summarizing and chat happen afterwards."""
    url = request.args.get("url", "").strip()

    if not is_youtube_url(url):
        return Response(sse("error", message="Please enter a valid YouTube URL."),
                        mimetype="text/event-stream", headers=SSE_HEADERS)

    @stream_with_context
    def generate():
        tmp = tempfile.mkdtemp(prefix="ytsum_")
        try:
            yield sse("status", stage="download", message="Downloading audio…")
            audio_path, title = download_audio(url, tmp)
            yield sse("title", title=title)

            yield sse("status", stage="transcribe",
                      message=f"Transcribing audio with Whisper ({WHISPER_MODEL})… this can take a bit.")
            segments = transcribe(audio_path)
            if not segments:
                yield sse("error", message="Could not transcribe any speech from this video.")
                return

            transcript = full_text(segments)
            duration = segments[-1]["end"]
            sid = new_session(title, segments, transcript)
            yield sse("transcript", text=transcript)
            yield sse("ready", sid=sid, title=title, duration=fmt_ts(duration),
                      words=len(transcript.split()),
                      chapters_available=duration >= CHAPTER_MIN_DURATION)
        except Exception as e:  # noqa: BLE001
            yield sse("error", message=f"{type(e).__name__}: {e}")
        finally:
            try:
                for f in os.listdir(tmp):
                    os.remove(os.path.join(tmp, f))
                os.rmdir(tmp)
            except OSError:
                pass

    return Response(generate(), mimetype="text/event-stream", headers=SSE_HEADERS)


@app.route("/summarize")
def summarize():
    """Summarize a transcript that /transcribe already put in a session."""
    sid = request.args.get("sid", "").strip()
    model = request.args.get("model", "llama3.2").strip() or "llama3.2"
    sess = SESSIONS.get(sid)

    if sess is None:
        return Response(sse("error", message="That transcript is no longer loaded. Transcribe the video again."),
                        mimetype="text/event-stream", headers=SSE_HEADERS)

    @stream_with_context
    def generate():
        try:
            yield sse("status", stage="summarize", message=f"Summarizing with {model}…")
            got_any = False
            for chunk in summarize_stream(sess["transcript"], model):
                got_any = True
                yield sse("summary_chunk", text=chunk)
            if not got_any:
                yield sse("error", message="The model returned no summary. Is the model pulled in Ollama?")
                return

            # Chapter breakdown for longer videos.
            segments = sess["segments"]
            if segments[-1]["end"] >= CHAPTER_MIN_DURATION:
                chapters = make_chapters(segments, CHAPTER_SECONDS)
                yield sse("chapters_start", count=len(chapters))
                for i, ch in enumerate(chapters):
                    yield sse("status", stage="chapters",
                              message=f"Building chapter {i + 1} of {len(chapters)}…")
                    raw = summarize_once(CHAPTER_PROMPT.format(transcript=ch["text"][:8000]), model)
                    title, summary = parse_chapter(raw)
                    yield sse("chapter", index=i, start=fmt_ts(ch["start"]),
                              end=fmt_ts(ch["end"]), title=title, summary=summary)

            yield sse("done", message="Complete.")
        except Exception as e:  # noqa: BLE001
            yield sse("error", message=f"{type(e).__name__}: {e}")

    return Response(generate(), mimetype="text/event-stream", headers=SSE_HEADERS)


@app.route("/chat", methods=["POST"])
def chat():
    """Answer a question about the loaded transcript, streaming plain text back."""
    body = request.get_json(silent=True) or {}
    sid = (body.get("sid") or "").strip()
    model = (body.get("model") or "llama3.2").strip() or "llama3.2"
    message = (body.get("message") or "").strip()
    sess = SESSIONS.get(sid)

    if sess is None:
        return {"error": "That transcript is no longer loaded. Transcribe the video again."}, 404
    if not message:
        return {"error": "Empty message."}, 400

    system = CHAT_SYSTEM.format(
        title=sess["title"], transcript=sess["transcript"][:CHAT_CONTEXT_CHARS]
    )
    history = sess["messages"][-CHAT_HISTORY_TURNS * 2:]
    messages = ([{"role": "system", "content": system}] + history
                + [{"role": "user", "content": message}])

    @stream_with_context
    def generate():
        reply = ""
        try:
            for chunk in chat_stream(model, messages):
                reply += chunk
                yield chunk
        except Exception as e:  # noqa: BLE001
            yield f"\n\n[{type(e).__name__}: {e}]"
            return
        # Only remember exchanges that actually completed.
        sess["messages"].append({"role": "user", "content": message})
        sess["messages"].append({"role": "assistant", "content": reply})

    return Response(generate(), mimetype="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  AV Digest running at  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
