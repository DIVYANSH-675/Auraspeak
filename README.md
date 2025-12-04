# 100x.inc | AuraSpeak 🧠🔊

A high-performance, ultra-low latency **API-first voice agent** designed for real-time interaction.
Built for **speed, scalability, and ambient computing** experiences.

Runs efficiently on a **standard laptop or cloud CPU** using the world's fastest APIs — no GPU required.

---

## 🚀 Try It Now (No Setup Required)

**Just click the link below to start talking:**

👉 **[Launch AuraSpeak Voice Bot](https://100x.up.railway.app)**



---

## ⚡ Key Features

* **Zero Setup Needed** — Works instantly in your browser.
* **Sub-500ms Latency** — It responds almost instantly, like a real person.
* **Smart Interruption** — You can interrupt it anytime, just like a real conversation.
* **High-Fidelity Audio** — Crystal clear voice quality.

---

## 🎮 How to Use

1. **Click the Link** above to open the web app.
2. **Tap the Glowing Orb** in the center.
3. **Allow Microphone Access** when asked.
4. **Speak Naturally** — Ask about my life, my superpower, or anything else!
5. **Interrupt Anytime** — If I'm talking too much, just speak over me.

---

## 📂 For Developers (Technical Details)

If you want to run this code yourself or understand how it works:

### **Core Backend**

| File                         | Role                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `server.py`                  | The FastAPI server. Manages WebSockets, routing, and audio buffers.               |
| `speech_pipeline_manager.py` | Orchestrates listening, thinking, and speaking using threads for low latency.     |
| `transcribe.py`              | Connects to Deepgram STT. Handles live transcription and end-of-speech detection. |
| `audio_in.py`                | Receives raw mic audio and prepares it safely for transcription.                  |
| `llm_module.py`              | LLM wrapper for Groq, OpenAI, MegaLLM with streaming text output.                 |
| `audio_module.py`            | Text-to-Speech module (Deepgram). Streams generated audio back to client.         |

### **Frontend (`/static`)**

| File                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `index.html`              | Main UI with grid layout + glowing core animation.           |
| `app.js`                  | Manages WebSockets, animations, audio context, and UI logic. |

---

## 🛠 Manual Installation (Optional)

**You do NOT need to install anything to use the bot.** But if you want to run the code locally:

### **1. Clone & Install**
```sh
# Clone the repo
git clone https://github.com/DIVYANSH-675/Auraspeak.git
cd Auraspeak

# Install dependencies
pip install -r requirements.txt
```

### **2. Run the Server**
```sh
python server.py
```

### **3. Open Browser**
Go to `http://localhost:8000`

---

## 👨‍💻 Developed By

**Divyansh Gupta**
🚀 Building the future of ambient AI.
