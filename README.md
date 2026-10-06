# Multimodal RAG (CLIP + BLIP)

A **Multimodal Retrieval-Augmented Generation (RAG)** system that combines **CLIP** for image-text understanding and **BLIP** for image captioning and visual understanding.

## Features

*  Image understanding with **CLIP**
*  Automatic image captioning with **BLIP**
*  Multimodal retrieval
*  RAG-based question answering
*  Simple GUI for adding images and querying the system

## Installation

```bash
pip install -r requirements.txt
```

Optionally configure your environment:

```bash
cp .env.example .env
```

## Run

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:8000
```

## Data

Add animal images to:

```text
data/images/
```

For example:

```text
data/images/
├── cat_1.jpg
├── lion_2.jpg
├── dog_1.jpg
└── tiger_1.jpg
```

You can also add images directly through the GUI.

Initial text data can be configured in:

```text
seed_texts.json
```

## Workflow

**Images → BLIP Captioning → CLIP Embeddings → Multimodal Retrieval → RAG → Answer**
