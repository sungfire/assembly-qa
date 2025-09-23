from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, firestore
import time

# Firebase 초기화
cred = credentials.Certificate("serviceAccountKey.json") 
firebase_admin.initialize_app(cred)
db = firestore.client()

# 모델 로딩
MODEL_PATH = "/workspace/2.AI학습모델파일/1. 질의응답/nia15-polyglot-5.8b-koalpaca-v1.1b-qna-best"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", dtype="auto"
)

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    # 사용자 입력 저장
    db.collection("chat_history").add({
        "role": "user",
        "message": query.message,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # 모델 추론
    inputs = tokenizer(query.message, return_tensors="pt").to("cuda")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 모델 답변 저장
    db.collection("chat_history").add({
        "role": "bot",
        "message": answer,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return {"answer": answer}

@app.get("/history")
def history():
    docs = db.collection("chat_history").order_by("timestamp").stream()
    return [{"role": d.to_dict()["role"], "message": d.to_dict()["message"]} for d in docs]
