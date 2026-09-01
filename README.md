# AV Digest

A small local web app: paste a YouTube link, and it downloads the audio and transcribes it
with **Whisper**. From there you can **summarize** it or **chat about it** with a **local
open-weight model via Ollama**, with the transcript held as context. Everything runs on
your machine. Summaries copy out as **Markdown** or export to **PDF**.

For longer videos (over 6 minutes by default) the summary also includes a **timestamped
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

Paste a YouTube URL and click **Transcribe**. Once the transcript is ready the view opens
into a workspace: controls on the left, chat in the middle, summary and chapters on the
right. Click **Summarize** when you want the summary, or just start asking questions.
**Copy Markdown** puts the summary and chapters on the clipboard as Markdown; **Save PDF**
writes them to a file.

Tick **Notify me when done** to get a browser notification when a transcript or summary
finishes while the tab is in the background.

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
| `OLLAMA_NUM_CTX`  | `8192`                   | Context window asked of Ollama. Its own default is 4096, which silently truncates long transcripts. |
| `SUMMARY_CONTEXT_CHARS` | `18000`            | Transcript characters sent when summarizing.     |
| `CHAT_CONTEXT_CHARS` | `12000`               | Transcript characters sent with each chat turn.  |
| `CHAT_HISTORY_TURNS` | `8`                   | Earlier exchanges replayed into a chat turn.     |
| `MAX_SESSIONS`    | `8`                      | Transcripts held in memory before the oldest is dropped. |

Example:

```bash
WHISPER_MODEL=small python app.py
```

## Notes

- Long videos take longer to transcribe. Start with a short clip to test.
- The transcript is trimmed before it goes to the model (see `SUMMARY_CONTEXT_CHARS` and
  `CHAT_CONTEXT_CHARS`) so it fits inside `OLLAMA_NUM_CTX`. Raising the character limits
  without raising `OLLAMA_NUM_CTX` means Ollama drops the overflow silently.
- Transcripts are kept in memory only. Restarting the server means transcribing again.
- Nothing is uploaded anywhere — download, transcription, summarizing, and chat are all
  local. The one exception is the jsPDF library, loaded from a CDN for PDF export.

## Troubleshooting

- **"No Ollama models detected"** — start Ollama and run `ollama pull llama3.2`, then reload.
- **ffmpeg errors on download** — confirm `ffmpeg -version` works in your terminal.
- **Slow transcription** — use a smaller `WHISPER_MODEL` (e.g. `tiny`) or a GPU.
