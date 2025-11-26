# 100x.inc | AuraSpeak 🧠🔊

A high-performance, ultra-low latency **API-first voice agent** designed for real-time interaction.
Built for **speed, scalability, and ambient computing** experiences.

Runs efficiently on a **standard laptop or cloud CPU** using the world's fastest APIs — no GPU required.

---

## ⚡ Key Features

* **Zero GPU Required** — Uses Deepgram (STT/TTS) + Groq/OpenAI (LLM).
* **Sub-500ms Latency** — Streaming pipeline starts speaking before generation finishes.
* **Smart Interruption** — Full-duplex mode: bot stops speaking when you talk.
* **High-Fidelity 48kHz Audio** — No upsampling artifacts, smooth streaming experience.

---

## 📂 File Structure

### **1. Core Backend**

| File                         | Role                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `server.py`                  | The FastAPI server. Manages WebSockets, routing, and audio buffers.               |
| `speech_pipeline_manager.py` | Orchestrates listening, thinking, and speaking using threads for low latency.     |
| `transcribe.py`              | Connects to Deepgram STT. Handles live transcription and end-of-speech detection. |
| `audio_in.py`                | Receives raw mic audio and prepares it safely for transcription.                  |
| `llm_module.py`              | LLM wrapper for Groq, OpenAI, MegaLLM with streaming text output.                 |
| `audio_module.py`            | Text-to-Speech module (Deepgram). Streams generated audio back to client.         |
| `text_context.py`            | Splits responses into natural sentence chunks for early speech.                   |
| `text_similarity.py`         | Prevents echo: filters out bot-generated speech inputs.                           |

---

### **2. Utilities**

| File                | Description                                 |
| ------------------- | ------------------------------------------- |
| `colors.py`         | ANSI color styling for readable logs.       |
| `logsetup.py`       | Central logging setup (timestamps, levels). |
| `requirements.txt`  | Lightweight dependency list.                |
| `environment.yml`   | Conda environment configuration.            |
| `start_unix.sh`     | One-click launcher for macOS/Linux.         |
| `start_windows.bat` | One-click launcher for Windows.             |

---

### **3. Frontend (`/static`)**

| File                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `index.html`              | Main UI with grid layout + glowing core animation.           |
| `app.js`                  | Manages WebSockets, animations, audio context, and UI logic. |
| `pcmWorkletProcessor.js`  | Captures raw mic input in a worker thread.                   |
| `ttsPlaybackProcessor.js` | Plays response audio smoothly with timing control.           |

---

## 🛠 Installation Guide

### **Prerequisites**

* API Keys:

  * `Deepgram` (Voice)
  * `Groq` or `OpenAI`
* Anaconda or Miniconda (recommended)

---

### 🚀 One-Click Install (Recommended)

#### **Windows**

Run:

```
start_windows.bat
```

#### **macOS / Linux**

```
chmod +x start_unix.sh
./start_unix.sh
```

---

### 🧰 Manual Installation

#### **Step 1 — Clean Environment**

```sh
# Optional: remove old venv
# Windows: rmdir /s /q venv
# Mac/Linux: rm -rf venv
```

#### **Step 2 — Choose Setup**

---

#### **Option A: Python venv**

```sh
python -m venv venv
```

Activate:

```sh
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install deps:

```sh
pip install -r requirements.txt
```

---

#### **Option B: Conda (Recommended)**

```sh
conda env create -f environment.yml
conda activate auraspeak
```

---

### **Step 3 — Configure `.env`**

Create `.env` or `.enve`:

```
DEEPGRAM_API_KEY=your_key
GROQ_API_KEY=your_groq_key
# OPENAI_API_KEY=optional

STT_BACKEND=deepgram
TTS_ENGINE=deepgram
LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192

DEEPGRAM_TTS_SAMPLE_RATE=48000
```

---

### **Step 4 — Run**

```sh
python server.py
```

Open browser:

👉 `http://localhost:8000`

---

## 🎮 How to Use

1. **Tap the core glow orb**
2. **Allow microphone access**
3. **Speak naturally**
4. **Interrupt anytime — it will stop and listen**
5. **Scroll chat history with ↑**
6. **Press PURGE to reset context**

---

## ⚠ Troubleshooting

| Issue            | Fix                                              |
| ---------------- | ------------------------------------------------ |
| Chipmunk voice   | Ensure `DEEPGRAM_TTS_SAMPLE_RATE=48000`          |
| No connection    | Check API keys and firewall (WebSocket required) |
| Mic not detected | Allow browser permissions / use localhost        |

---

## 👨‍💻 Developed By

**Divyansh Gupta**
🚀 Building the future of ambient AI.

---
