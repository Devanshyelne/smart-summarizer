# ✨ T5 Text Summarizer

A full-stack text and dialogue summarization app powered by a fine-tuned **T5-small** transformer model, served through a **FastAPI** backend with a clean, minimal **Tailwind CSS** frontend.

Paste in an article, email, meeting notes, or conversation — get a concise, coherent summary in seconds.

---

## 🧠 Overview

This project covers the complete machine learning pipeline — from raw data preprocessing to a deployed, working web application:

1. **Data preprocessing & cleaning** — text normalization, whitespace/HTML stripping
2. **Tokenization** — using the T5 tokenizer with padding/truncation to fixed sequence lengths
3. **Fine-tuning** — T5-small fine-tuned on the **SAMSum dataset** (dialogue-summary pairs) using Hugging Face's `Trainer` API
4. **Inference pipeline** — beam search decoding for high-quality summary generation
5. **Deployment** — wrapped in a FastAPI REST API with an interactive frontend

The entire training process — from raw CSV to a saved model — is documented step-by-step in the included Jupyter notebook.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Model | T5-small (Text-to-Text Transfer Transformer) |
| Training | Hugging Face `transformers`, PyTorch |
| Backend | FastAPI, Jinja2 |
| Frontend | HTML, Tailwind CSS, Vanilla JS (async/fetch) |
| Server | Uvicorn (ASGI) |
| Dataset | SAMSum (dialogue summarization corpus) |

---

## 📂 Project Structure
## 📂 Project Structure

```
├── app.py                    # FastAPI backend + inference logic
├── index.html                 # Frontend UI
├── Text_Summarizer.ipynb      # Full training pipeline (data → fine-tuned model)
├── .gitignore
└── README.md
```

> Note: `saved_summary_model/` (trained weights, ~230MB) is excluded from this repo due to GitHub's file size limits. See **Setup** below to generate it locally.

---

## ⚙️ How It Works

1. **Input** — user pastes text into the frontend textarea
2. **Preprocessing** — text is cleaned (whitespace normalized, HTML stripped, lowercased) to match training-time preprocessing
3. **Tokenization** — input is tokenized and padded to a fixed length of 512 tokens
4. **Generation** — the fine-tuned T5 model generates a summary using **beam search** (`num_beams=4`) with early stopping, capped at 150 tokens
5. **Decoding** — output token IDs are decoded back into readable text and returned to the frontend

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Devanshyelne/t5-text-summarizer.git
cd t5-text-summarizer
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn transformers torch pandas jinja2
```

### 3. Train the model
Open and run `Text_Summarizer.ipynb` end-to-end. This will:
- Load and preprocess the SAMSum dataset
- Fine-tune T5-small for several epochs
- Save the trained model + tokenizer to `./saved_summary_model/`

> Training uses GPU (CUDA) or Apple Silicon (MPS) automatically if available, falling back to CPU otherwise.

### 4. Run the server
```bash
uvicorn app:app --reload
```

### 5. Open the app
Navigate to `http://127.0.0.1:8000` in your browser and start summarizing.

---

## 🔌 API Reference

**POST** `/summarize/`

Request body:
```json
{
  "dialogue": "Your text or conversation here..."
}
```

Response:
```json
"Generated summary text..."
```

---

## 🎯 Model Details

- **Base model:** T5-small (60M parameters)
- **Fine-tuning dataset:** SAMSum — real-life style messenger conversations paired with human-written summaries
- **Training config:** 6 epochs, batch size 8, warmup steps 500, weight decay 0.01
- **Decoding strategy:** Beam search (4 beams), max output length 150 tokens

---

## 🔮 Future Improvements

- [ ] Support for longer documents via chunking + hierarchical summarization
- [ ] Add extractive + abstractive summary toggle
- [ ] Deploy backend to a cloud service (Render/Railway) with model hosted on Hugging Face Hub
- [ ] Add proper sentence casing/punctuation post-processing on output
- [ ] Dockerize for easier setup

---

## 👤 Author

**Devansh Yelne**
[GitHub](https://github.com/Devanshyelne)
