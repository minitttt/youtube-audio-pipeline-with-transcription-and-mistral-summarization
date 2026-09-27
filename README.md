# 🎬 AI Video Assistant

Turn any YouTube video (or local audio/video file) into a searchable, chattable meeting brief — transcript, summary, action items, key decisions, and a RAG-powered chatbot to ask follow-up questions, all in one Streamlit app.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Mistral AI](https://img.shields.io/badge/LLM-Mistral%20AI-orange)
![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-lightgrey)

---

## ✨ Features

- **🔊 Universal audio ingestion** — paste a YouTube URL or point to a local audio/video file
- **📝 Automatic transcription** — powered by OpenAI Whisper (runs locally, no external API needed)
- **📌 Auto-generated title** — a short, professional title summarizing the content
- **📋 Smart summarization** — map-reduce summarization pipeline (handles long transcripts by chunking, summarizing each piece, then combining)
- **✅ Structured extraction** — automatically pulls out:
  - Action items (task, owner, deadline)
  - Key decisions made
  - Open / unresolved questions
- **🧠 RAG-powered chat** — ask natural-language questions about the transcript and get grounded answers via a vector-search + LLM pipeline (Chroma + HuggingFace embeddings + Mistral)
- **🎨 Polished dark-mode UI** — built with Streamlit, custom CSS, live pipeline status indicators in the sidebar

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
   YouTube URL /    │  audio_processor.py │  → yt-dlp download → pydub
   local file        │  (Audio Ingestion)   │    convert to WAV → chunk
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   transcribe.py      │  → Whisper (local ASR)
                    │  (Transcription)     │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
        ┌───────────────┐ ┌──────────┐ ┌─────────────────┐
        │  summary.py    │ │extractor │ │  vector_store.py │
        │  (Title +      │ │  .py     │ │  (Chroma +       │
        │  Summary)      │ │(Action   │ │  Embeddings)     │
        │                │ │ items,   │ │                  │
        │                │ │decisions,│ │                  │
        │                │ │questions)│ │                  │
        └───────────────┘ └──────────┘ └────────┬─────────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │  rag_engine.py      │
                                        │  (Retrieval +        │
                                        │  Mistral Chat)        │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌───────────────────┐
                                        │     app.py          │
                                        │  (Streamlit UI)      │
                                        └───────────────────┘
```

All LLM calls (summarization, extraction, RAG chat) run through **Mistral AI** via LangChain (`langchain_mistralai`).

---

## 📁 Project Structure

```
video_agent/
├── app.py                  # Streamlit UI — main entry point
├── utlis/
│   └── audio_processor.py  # YouTube download, WAV conversion, chunking
├── core/
│   ├── transcribe.py       # Whisper-based transcription
│   ├── summary.py          # Title generation + map-reduce summarization
│   ├── extractor.py        # Action items / decisions / questions extraction
│   ├── rag_engine.py       # RAG chain for Q&A over transcript
│   └── vector_store.py     # Chroma vector store + embeddings
├── .env                    # API keys (never committed — see .gitignore)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Prerequisites

- **Python 3.9+**
- **FFmpeg** installed and available on your system PATH (required by `pydub` and `yt-dlp` for audio extraction/conversion)
  ```bash
  brew install ffmpeg        # macOS
  ```
- A **Mistral AI API key** — get one at [console.mistral.ai](https://console.mistral.ai/api-keys)

---

## 🚀 Setup

**1. Clone the repo**
```bash
git clone https://github.com/minitttt/youtube-audio-pipeline-with-transcription-and-mistral-summarization.git
cd youtube-audio-pipeline-with-transcription-and-mistral-summarization
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key_here
WHISPER_MODEL=small
```
> ⚠️ Do not wrap the key in quotes — `.env` files treat quotes as literal characters, which will break the API key.

**5. Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. Open the sidebar and paste a **YouTube URL** or a **local file path**
2. Select the language (`english` or `hinglish`)
3. Click **⚡ Analyse**
4. Watch the live pipeline status (audio → transcript → title → summary → extraction → RAG) in the sidebar
5. Once complete, review the **title, summary, action items, key decisions, and open questions**
6. Scroll down to **chat with the transcript** — ask any question and get grounded, context-aware answers

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Audio download | yt-dlp |
| Audio processing | pydub |
| Transcription | OpenAI Whisper (local) |
| LLM orchestration | LangChain |
| LLM provider | Mistral AI (`mistral-small-latest`) |
| Vector store | Chroma |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |

---

## ⚠️ Known Limitations

- **YouTube download reliability**: YouTube frequently changes its anti-bot measures; if downloads start failing with `403 Forbidden` or `VideoUnavailable` errors, update `yt-dlp` (`pip install -U yt-dlp`) — this is usually enough to resolve it.
- **Whisper performance**: transcription speed depends on the model size (`tiny`/`base`/`small`/`medium`/`large`) and available CPU/GPU. The `small` model is a good balance for most machines.
- **Long videos**: very long videos are chunked in 10-minute segments for transcription and summarized via map-reduce to stay within LLM context limits.

---



## 📄 License

This project is open for personal and educational use. Add a license of your choice (MIT recommended) if you plan to open-source it publicly.
