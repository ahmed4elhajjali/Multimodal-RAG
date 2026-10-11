# Multimodal RAG (CLIP + BLIP)

A **Multimodal Retrieval-Augmented Generation (RAG)** system that combines **CLIP** for cross-modal image-text embeddings and **BLIP** for automated visual captioning and semantic retrieval.

---

## 🌟 Features

* **Cross-Modal Retrieval with CLIP (`openai/clip-vit-base-patch32`)**: Retrieve facts and images using joint embedding spaces.
* **Automated Image Captioning with BLIP (`Salesforce/blip-image-captioning-base`)**: Generate descriptive captions on upload to enrich searchability.
* **Hybrid Visual Scoring**: Balances direct image-text similarity and caption-text similarity with tunable $\alpha$ parameter.
* **Modern Web Interface**: Responsive dashboard with real-time similarity metrics, preset prompt chips, image dropzone, and system telemetry.
* **Knowledge Base Management**: Full CRUD support for text knowledge facts and image catalog, plus one-click reset to seed data and JSON export.
* **System Telemetry**: Real-time CUDA / CPU device status, vector dimensions, and model configurations.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ahmed4elhajjali/Multimodal-RAG.git
cd Multimodal-RAG
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

Create your `.env` configuration file:

```bash
cp .env.example .env
```

Available environment variables:
| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Hugging Face model identifier for CLIP |
| `BLIP_MODEL` | `Salesforce/blip-image-captioning-base` | Hugging Face model identifier for BLIP |
| `DEVICE` | `cuda` (if available) / `cpu` | Inference acceleration device |
| `TOP_K` | `3` | Default number of top results to retrieve |
| `ALPHA` | `0.6` | Image similarity vs caption similarity weight |
| `LOGIT_SCALE`| `50` | Softmax temperature for relevance percentage |
| `HOST` | `127.0.0.1` | Server host address |
| `PORT` | `8000` | Server port |

### 3. Run the Application

```bash
python app.py
```

The interactive web dashboard will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation (Swagger) is available at:
```text
http://127.0.0.1:8000/docs
```

---

## 📁 Project Structure

```text
Multimodal-RAG/
├── app.py              # FastAPI server & REST API endpoints
├── config.py           # Configuration management & environment loading
├── models.py           # CLIP and BLIP model loading & inference wrapper
├── rag.py              # Multimodal RAG retrieval and scoring logic
├── store.py            # In-memory vector store & persistence (kb.json)
├── seed_texts.json     # Initial default knowledge facts
├── requirements.txt    # Project dependencies
├── .env.example        # Example environment variables
├── .gitignore          # Git exclusion rules
├── data/
│   ├── kb.json         # Persistent knowledge base state
│   └── images/         # Stored knowledge base images
└── static/
    └── index.html      # Single-page interactive frontend
```

---

## 🔄 Workflow

```
Query (Text / Image)
        │
        ▼
   CLIP Encoder ────────┐
        │               │
        ▼               ▼
Multimodal Search ◄──── BLIP Captioning
        │
        ▼
  Ranked Results (Relevance Scores & Retrieved Context)
```

---

## 📜 License

MIT License. Feel free to use and contribute!
