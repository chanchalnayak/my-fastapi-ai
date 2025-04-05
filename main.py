from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn


app = FastAPI()

# ✅ Load AI Model (distilgpt2)
try:
    ai_model = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    print("Error loading model:", str(e))
    ai_model = None

# ✅ Request Body Model
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "AI API is running with distilgpt2!"}

@app.post("/chat/")
def chat(request: ChatRequest):
    if not ai_model:
        raise HTTPException(status_code=500, detail="AI model not loaded")
    
    response = ai_model(request.prompt, max_length=50, do_sample=True)[0]['generated_text']
    return {"response": response}

# ✅ For Render: Proper Port Binding
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)