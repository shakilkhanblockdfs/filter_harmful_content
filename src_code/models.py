# models.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class TextModerationModel:
    def __init__(self, model_name="unitary/toxic-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Labels depend on model; this is a typical toxic classifier
        self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    @torch.no_grad()
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}

        # Aggregate risk score (simple max, you can do weighted sum)
        risk_score = float(np.max(probs))

        return {
            "scores": scores,
            "risk_score": risk_score
        }
