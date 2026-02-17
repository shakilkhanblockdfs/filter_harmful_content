📁 1. Project Structure
Create a folder and files like this:
content_moderation/
├── app.py
├── models.py
├── policy.py
├── schemas.py
├── train.py
├── requirements.txt
└── README.md   (optional)
 
🧰 2. Prerequisites
Make sure you have:
•	Python 3.9+ installed
Check:
python --version
•	pip installed:
pip --version
(Optional but recommended)
•	Create a virtual environment:
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
 
📦 3. Install Dependencies
Create requirements.txt:
fastapi
uvicorn
torch
transformers
pydantic
numpy
datasets
Install everything:
pip install -r requirements.txt
Verify install:
python -c "import torch, transformers, fastapi; print('All good!')"
 
🧠 4. Downloading the Model (Automatic)
The first time you run the API, HuggingFace will automatically download:
unitary/toxic-bert
No manual step needed. It will cache it locally.
 
🚀 5. Run the API Server
From inside content_moderation/:
uvicorn app:app --reload
You should see something like:
Uvicorn running on http://127.0.0.1:8000
Open in browser:
•	API docs: 👉 http://127.0.0.1:8000/docs
•	Health check: 👉 http://127.0.0.1:8000/health
 
🧪 6. Test the API
Option A: Using Swagger UI (Easiest)
1.	Go to: http://127.0.0.1:8000/docs
2.	Click POST /moderate
3.	Click “Try it out”
4.	Paste:
{
  "text": "I hate you and I will hurt you",
  "user_id": "user123"
}
5.	Click “Execute”
You’ll get a response like:
{
  "action": "LIMIT_AND_REVIEW",
  "risk_score": 0.91,
  "scores": {
    "toxic": 0.93,
    "severe_toxic": 0.12,
    "obscene": 0.03,
    "threat": 0.88,
    "insult": 0.79,
    "identity_hate": 0.01
  }
}
 
Option B: Using curl
curl -X POST "http://127.0.0.1:8000/moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "You are stupid and I hate you",
    "user_id": "user42"
  }'
 
⚙️ 7. How the Decision Works
In policy.py:
if risk_score > 0.95:
    BLOCK
elif risk_score > 0.70:
    LIMIT_AND_REVIEW
elif risk_score > 0.40:
    WARN
else:
    ALLOW
You can tune these thresholds based on:
•	Business rules
•	False positives
•	Legal/safety requirements
 
🏋️ 8. Training Your Own Model (Optional but Realistic)
The file train.py is a starter training script.
Run:
python train.py
What it does:
•	Loads a dataset (currently IMDB as placeholder)
•	Fine-tunes a BERT-like model
•	Saves it to:
./moderation_model/
To use your own trained model:
In models.py, change:
model_name = "unitary/toxic-bert"
to:
model_name = "./moderation_model"
Then restart the API:
uvicorn app:app --reload

<img width="468" height="644" alt="image" src="https://github.com/user-attachments/assets/ef71d6e8-a984-434e-99b4-d76bac69b52e" />
