# 🧠  AI – Final Therapy Assistant

Welcome to **AI**, an intelligent therapy assistant designed to deliver empathetic, context-aware, and voice-enabled mental health support. It uses **OpenAI GPT-4o** for conversational therapy, **ElevenLabs** for multilingual voice synthesis, and PDF knowledge embedding for therapeutic context.

## 🚀 Features

- 🗣️ **Conversational Therapy AI** – Empathetic, multilingual chatbot based on GPT-4o.
- 📚 **PDF Knowledge Base** – Embed and retrieve therapy-related content using vector search.
- 🔍 **Crisis Detection** – Automatically detects critical language (e.g., self-harm) and responds appropriately.
- 🔊 **Text-to-Speech (TTS)** – Voice output using ElevenLabs voices.
- 🧠 **Therapy Types** – Supports CBT, DBT, Grief, Anxiety, Depression, Parenting, and more.
- 🎤 **Speech-to-Text (STT)** – Transcribes user voice input using OpenAI Whisper.
- 📊 **Streamlit Dashboard** – Interactive testing panel with analytics, session history, and automated tests.
- ✅ **Automated Testing** – Five built-in therapy test cases for QA and validation.

## 🗂️ Project Structure

```

final\_therapy/
├── pdf/                   # Folder containing source PDFs
├── vector\_store/          # Folder to store vector DB files
├── main.py                # Core AI logic and orchestration
├── test\_dashboard.py      # Streamlit UI for testing AI
├── prompt.py              # Prompt templates and crisis detection
├── pdf\_processor.py       # Handles PDF parsing and vector store building
├── test.py                # Test script or standalone test runner
├── requirements.txt       # Required Python libraries
└── .env                   # API keys and environment configs (not committed)

````

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/final_therapy.git
cd final_therapy
````

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

```env
OPENAI_API_KEY=your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key
PDF_FOLDER_PATH=./pdf/
VECTOR_STORE_PATH=./vector_store/
OPENAI_MODEL=gpt-4o
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

### 5. Add Therapy PDFs

Place your PDF therapy documents in the `./pdf/` folder. These will be embedded into the vector database.

## 🧪 How to Run

### 1. Run the AI Backend (for API or CLI)

```bash
python main.py
```

### 2. Run the Streamlit Dashboard (Recommended)

```bash
streamlit run test_dashboard.py
```

This will launch an interactive web interface for:

* Chatting with the AI
* Recording voice and receiving audio replies
* Running automated therapy tests
* Viewing session analytics and vector stats

## 🧠 Therapy Types Supported

```python
from prompt import TherapyType

TherapyType.ANXIETY
TherapyType.DEPRESSION
TherapyType.PARENTING
TherapyType.GRIEF
TherapyType.GENERAL
TherapyType.CBT
TherapyType.DBT
# and more...
```

## 🛡️ Crisis Mode

If the AI detects potentially harmful statements (e.g. *"I want to hurt myself"*), it will:

* Skip normal context generation
* Respond with a pre-designed crisis message including helpline info
## 📈 Dashboard Tabs Overview

| Tab                | Description                  |
| ------------------ | ---------------------------- |
| 💬 Chat Interface  | Test real conversations      |
| 🧪 Automated Tests | Run 5+ predefined test cases |
| 🎤 Voice Testing   | Record and test with audio   |
| 📊 Analytics       | See system stats & metrics   |
| 📋 Session History | View past conversation logs  |

## 📦 Export & Session Management

* Export conversations in JSON
* Reset sessions
* Track PDF usage and therapy type trends

## 📸 Screenshots

> You can upload screenshots to this section to showcase your UI and interactions.

---





