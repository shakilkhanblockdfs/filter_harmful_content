# app.py
from fastapi import FastAPI
from schemas import ModerateRequest, ModerateResponse
from models import TextModerationModel, ImageModerationModel
from policy import decide_action
import queue
import threading
from fastapi import UploadFile
from PIL import Image
import json
from sqlalchemy import create_engine, Column, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="Content Moderation API")

model = TextModerationModel()
image_model = ImageModerationModel()

# Simulated human review queue
review_queue = queue.Queue()

def human_review_worker():
    while True:
        item = review_queue.get()
        if item is None:
            break
        print("🧑‍⚖️ Human review needed for:", item)

threading.Thread(target=human_review_worker, daemon=True).start()

DATABASE_URL = "sqlite:///./moderation.db"  # Replace with Postgres/Mongo URL
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    text = Column(String)
    risk_score = Column(Float)
    scores = Column(JSON)
    action = Column(String)

Base.metadata.create_all(bind=engine)

@app.post("/moderate", response_model=ModerateResponse)
def moderate(req: ModerateRequest):
    result = model.predict(req.text)

    risk_score = result["risk_score"]
    scores = result["scores"]

    action = decide_action(risk_score)

    # Send to human review if needed
    if action in ["LIMIT_AND_REVIEW", "BLOCK"]:
        review_queue.put({
            "user_id": req.user_id,
            "text": req.text,
            "scores": scores,
            "risk_score": risk_score
        })

    return ModerateResponse(
        action=action,
        risk_score=risk_score,
        scores=scores
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/moderate_image")
def moderate_image(file: UploadFile):
    image = Image.open(file.file)
    text_prompts = ["safe", "unsafe"]
    probs = image_model.predict(image, text_prompts)
    return {"safe": float(probs[0][0]), "unsafe": float(probs[0][1])}

@app.post("/log_moderation")
def log_moderation(req: ModerateRequest):
    db = SessionLocal()
    result = model.predict(req.text)
    risk_score = result["risk_score"]
    scores = result["scores"]
    action = decide_action(risk_score)

    log = ModerationLog(
        id=req.user_id,
        user_id=req.user_id,
        text=req.text,
        risk_score=risk_score,
        scores=scores,
        action=action
    )
    db.add(log)
    db.commit()
    db.close()

    return {"status": "logged"}
