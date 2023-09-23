from fastapi import FastAPI, Depends
from pydantic import BaseModel
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import ray

class Item(BaseModel):
    text: str

app = FastAPI()

ray.init()
model_path = 'https://drive.google.com/file/d/12gPEPM-a2lzwBSWLbT-wWTjjr_Zd14kK/view?usp=sharing'  # replace with the path to your local model file

@ray.remote
def load_model(model_path):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

model_id = load_model.remote(model_path)
model = ray.get(model_id)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model.eval()

@app.on_event("startup")
async def load_model_on_startup():
    global model
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@app.post("/predict")
async def predict(item: Item):
    global model
    test_input = [item.text]
    tokenized_test_input = tokenizer(test_input, truncation=True, padding=True, return_tensors="pt")

    tokenized_test_input = tokenized_test_input.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    output_tokens = model.generate(tokenized_test_input.input_ids, num_beams=5, temperature=1.5, repetition_penalty=1.2)
    output_text = [tokenizer.decode(tokens) for tokens in output_tokens]

    return {"prediction": output_text}
