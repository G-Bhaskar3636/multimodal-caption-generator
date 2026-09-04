# 🖼️ Multimodal Caption Generator

An image captioning and storytelling application with multilingual output, semantic search over history, and an agentic chatbot — built to demonstrate applied **Data Science**, **GenAI**, and **Agentic AI** skills.

## Features

- **Image Captioning & Story Generation** — upload an image, get an AI-generated caption and short story, in your choice of language (Gemini for vision + generation)
- **Semantic Search over History** — every caption is embedded and stored; search past uploads by meaning, not just keywords (RAG with ChromaDB + sentence-transformers)
- **Agentic Chatbot** — a general-purpose chatbot (Groq) that optionally unlocks image-analysis tools (captioning, object detection) when an image is provided, decides on its own which tool to call, and remembers prior conversation turns via a separate memory store

## Architecture

```
app/
├── agent/
│   ├── tools.py         # Tools available to the agent (caption, object detection)
│   ├── memory.py         # Conversation memory — own persistent store (agent_chroma_store)
│   └── ai_agent.py       # Orchestrator: Tools + RAG + Memory → LLM → Final Answer
├── rag/
│   ├── embedder.py       # Text embedding (sentence-transformers)
│   ├── vector_store.py   # RAG's caption history store (chroma_store)
│   └── rag_manager.py    # Public interface for storing/searching captions
├── pages/
│   ├── home.py
│   ├── image_section.py  # Captioning + story generation (Gemini)
│   ├── translation.py    # Translation helper
│   ├── search_history.py # Semantic search UI
│   └── agent.py          # Chatbot UI (Groq agent)
└── app.py                # Streamlit multipage entry point
```

**Agent flow:**

```
USER → ai_agent.py → AI AGENT
                        ├── Tools (image captioning, object detection — optional)
                        ├── RAG (search past captions, chroma_store)
                        └── Memory (recall past conversation, agent_chroma_store)
                                │
                        Tool Results + Context → LLM (Groq) → Final Answer
```

## Tech Stack

| Area | Tools |
|---|---|
| App framework | Streamlit (multipage) |
| Image captioning / vision | Google Gemini (`gemini-2.5-flash`) |
| Object detection | Hugging Face `transformers` (DETR-ResNet-101) |
| Agent reasoning / tool-calling | Groq (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector storage (RAG + Memory) | ChromaDB (two separate persistent collections) |
| Translation | NLLB-200 / Gemini |

## Setup

### 1. Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd multimodal-caption-generator
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file in the project root (same folder as `app/`) with:
```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```
- Get a Gemini key: https://ai.google.dev/gemini-api/docs/api-key
- Get a Groq key: https://console.groq.com/keys

### 4. Run the app
```bash
cd app
streamlit run app.py
```

## Usage

| Page | What it does |
|---|---|
| **Image Captioning** | Upload an image, pick a language, get a caption + short story |
| **Search History** | Semantically search past generated captions |
| **AI Chatbot** | Chat freely; optionally upload an image to unlock captioning/object-detection tools mid-conversation |

## Notes on Model Choices

- Groq's model lineup changes over time — `openai/gpt-oss-20b` is used here as it's on the accessible developer tier as of this writing. Check `console.groq.com/docs/models` if you hit a `model_not_found` error, and update `GROQ_MODEL` in `app/agent/ai_agent.py`.
- The agent's tool-calling loop is implemented manually (Groq's OpenAI-compatible API doesn't auto-execute functions like some SDKs do) — see `ai_agent.py` for the request → tool execution → follow-up request pattern.

## Possible Extensions

- Extend NLLB translation coverage to all supported Indian languages (currently English/Hindi/Telugu)
- Add evaluation metrics (BLEU/CIDEr) against a labeled captioning dataset
- Persist chat sessions across browser refreshes (currently session-scoped via `st.session_state`)
- Dockerize + deploy (Hugging Face Spaces / Render / Streamlit Cloud)

## License

MIT