from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, firestore

# 🔸 Firebase 초기화
cred = credentials.Certificate("serviceAccountKey.json")  # Firebase 콘솔에서 발급
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🔸 모델 로딩
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

class ChatRequest(BaseModel):
    uid: Optional[str] = None
    message: str
    sessionId: Optional[str] = None
    sessionTitle: Optional[str] = None

def _session_collection(uid: str):
    return db.collection("users").document(uid).collection("sessions")


def _generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _serialize_timestamp(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return None


@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    if not request.uid:
        answer = _generate_answer(message)
        history_ref = db.collection("chat_history")
        history_ref.add(
            {
                "role": "user",
                "message": message,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
        history_ref.add(
            {
                "role": "bot",
                "message": answer,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
        return {"answer": answer}

    sessions_ref = _session_collection(request.uid)
    session_title = request.sessionTitle
    session_doc_ref = None
    session_id = request.sessionId

    if session_id:
        session_doc_ref = sessions_ref.document(session_id)
        snapshot = session_doc_ref.get()
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="Session not found.")
        existing_title = snapshot.to_dict().get("title")
        if not session_title:
            session_title = existing_title
    else:
        # 새 세션 생성
        if not session_title:
            session_title = message[:20] + ("..." if len(message) > 20 else "")
        if not session_title:
            session_title = "새 채팅"
        session_doc_ref = sessions_ref.document()
        session_doc_ref.set(
            {
                "title": session_title,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
        )
        session_id = session_doc_ref.id

    if session_doc_ref is None:
        session_doc_ref = sessions_ref.document(session_id)

    messages_ref = session_doc_ref.collection("messages")

    messages_ref.add(
        {
            "role": "user",
            "text": message,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )

    answer = _generate_answer(message)

    messages_ref.add(
        {
            "role": "bot",
            "text": answer,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )

    update_payload = {"updatedAt": firestore.SERVER_TIMESTAMP}
    if session_title:
        update_payload["title"] = session_title
    session_doc_ref.update(update_payload)

    return {"answer": answer, "sessionId": session_id, "title": session_title}


@app.get("/api/users/{uid}/sessions")
def get_sessions(uid: str):
    if not uid:
        raise HTTPException(status_code=400, detail="uid is required.")
    sessions_ref = _session_collection(uid)
    docs = sessions_ref.order_by("updatedAt", direction=firestore.Query.DESCENDING).stream()
    sessions = []
    for doc_snapshot in docs:
        data = doc_snapshot.to_dict() or {}
        sessions.append(
            {
                "id": doc_snapshot.id,
                "title": data.get("title", "이전 채팅"),
                "createdAt": _serialize_timestamp(data.get("createdAt")),
                "updatedAt": _serialize_timestamp(data.get("updatedAt")),
            }
        )
    return {"sessions": sessions}


@app.get("/api/users/{uid}/sessions/{session_id}/messages")
def get_session_messages(uid: str, session_id: str):
    if not uid or not session_id:
        raise HTTPException(status_code=400, detail="uid and session_id are required.")
    session_doc_ref = _session_collection(uid).document(session_id)
    if not session_doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages_ref = session_doc_ref.collection("messages")
    docs = messages_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).stream()
    messages = []
    for doc_snapshot in docs:
        data = doc_snapshot.to_dict() or {}
        messages.append(
            {
                "id": doc_snapshot.id,
                "role": data.get("role"),
                "text": data.get("text"),
                "timestamp": _serialize_timestamp(data.get("timestamp")),
            }
        )
    return {"messages": messages}
