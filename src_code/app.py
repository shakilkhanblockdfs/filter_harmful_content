# app.py
from fastapi import FastAPI
from schemas import ModerateRequest, ModerateResponse
from models import TextModerationModel
from policy import decide_action
import queue
import threading

app = FastAPI(title="Content Moderation API")

model = TextModerationModel()

# Simulated human review queue
review_queue = queue.Queue()

def human_review_worker():
    while True:
        item = review_queue.get()
        if item is None:
            break
        print("🧑‍⚖️ Human review needed for:", item)

threading.Thread(target=human_review_worker, daemon=True).start()

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
