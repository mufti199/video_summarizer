# YouTube Audio Summarizer

A small local web app: paste a YouTube link, and it downloads the audio, transcribes it
with **Whisper**, and summarizes it with a **local open-weight model via Ollama**.
Everything runs on your machine. Summaries export to **PDF** or **.txt**.

For longer videos (over 6 minutes by default) it also produces a **timestamped
chapter breakdown** — each ~5-minute section gets its own title and short summary,
included in the on-screen view and both exports.

## What you need

1. **Python 3.10+**
2. **ffmpeg** (used to extract audio)
   - macOS: `brew install ffmpeg`
   - Windows: `winget install Gyan.FFmpeg` (or download from ffmpeg.org)
   - Linux: `sudo apt install ffmpeg`
3. **Ollama** running locally — https://ollama.com
   - After installing, pull a model, e.g. `ollama pull llama3.2`
   - Make sure Ollama is running (the app talks to it at `http://localhost:11434`).

## Setup

```bash
pip install -r requirements.txt
```

The first run also downloads the Whisper model weights automatically (a few hundred MB
for the default `base` model).

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

Paste a YouTube URL, pick your Ollama model, and click **Summarize**. When it's done,
use **Save PDF**, **Save .txt**, or **Copy** to share the result.

## Configuration (optional environment variables)

| Variable          | Default                  | Notes                                             |
|-------------------|--------------------------|---------------------------------------------------|
| `WHISPER_MODEL`   | `base`                   | `tiny`/`base`/`small`/`medium`/`large-v3`. Bigger = more accurate, slower. |
| `WHISPER_DEVICE`  | `cpu`                    | Set to `cuda` if you have an NVIDIA GPU.          |
| `WHISPER_COMPUTE` | `int8`                   | Use `float16` on GPU for best quality.           |
| `OLLAMA_HOST`     | `http://localhost:11434` | Change if Ollama runs elsewhere.                 |
| `PORT`            | `5000`                   | Web server port.                                 |
| `CHAPTER_SECONDS` | `300`                    | Length of each timestamped chapter window (secs).|
| `CHAPTER_MIN_DURATION` | `360`               | Only build chapters for videos longer than this. |

Example:

```bash
WHISPER_MODEL=small python app.py
```

## Notes

- Long videos take longer to transcribe. Start with a short clip to test.
- The transcript is trimmed to ~24k characters before summarizing so it fits the model's
  context. For very long videos, consider a larger-context model in Ollama.
- Nothing is uploaded anywhere — download, transcription, and summarization are all local.

## Troubleshooting

- **"No Ollama models detected"** — start Ollama and run `ollama pull llama3.2`, then reload.
- **ffmpeg errors on download** — confirm `ffmpeg -version` works in your terminal.
- **Slow transcription** — use a smaller `WHISPER_MODEL` (e.g. `tiny`) or a GPU.
