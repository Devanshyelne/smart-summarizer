# fastapi --> python based web framwork
# API -- Application Programming Interface
# cilent send request which API recives and pass on 
# to the server then API gives back response to cilent 

#uvicorn - (lightweight web server)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch 
import re 
from fastapi.templating import Jinja2Templates # to show UI part
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer APP", description="Text Summarization using T5", version="1.0")

# model & tokenzier 
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating 
templates = Jinja2Templates(directory=".") # . --> dot matable ussi directory mai available hai

# Input Schema for dialogue # Schema --> matable fromat 
class DialogueInput(BaseModel):
    dialogue: str # it tell ki API string format ko expect kar raha hia

def clean_data(text):
    text = re.sub(r"\r\n", " " , text) # remove lines 
    text = re.sub(r"\s+" , " " , text) # remove spaces
    text = re.sub(r"<.*?>", " ", text) # remove html tags like <p> <h1>
    text = text.strip().lower() #aage phiche wali extra space of hatake sab ko lower mai convert karta hai
    return text


def summarize_dialogue(dialogue : str) -> str: # basciilay jo dialogue hai to string type ka hai and output bi string hi chahiye
    dialogue = clean_data(dialogue) #clean 

    # tokenize 
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    # token ids convert to summary => decoding // decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens = True) #EOS , separators
    return summary

# Add Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return summary

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")



