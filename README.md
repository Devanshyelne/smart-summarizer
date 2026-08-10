# Text Summarizer (T5 + FastAPI)

A dialogue/text summarization app fine-tuned on the SAMSum dataset using T5-small, served via FastAPI with a Tailwind frontend.

## Structure
- `Text_Summarizer.ipynb` — data preprocessing, tokenization, fine-tuning T5 on SAMSum
- `app.py` — FastAPI backend serving the trained model
- `index.html` — frontend UI

## Setup
1. Install dependencies:
   pip install fastapi uvicorn transformers torch pandas jinja2

2. Run the notebook to train and save the model to `./saved_summary_model/`
   (not included in repo due to size — ~230MB)

3. Start the server:
   uvicorn app:app --reload

4. Open `http://127.0.0.1:8000` in your browser.