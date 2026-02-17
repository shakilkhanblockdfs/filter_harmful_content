# models.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel, BertModel, BertTokenizer, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import numpy as np
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from cryptography.fernet import Fernet

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        return ' '.join(filtered_words)

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

class ImageModerationModel:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image, text_prompts):
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
        return probs

class EmbeddingGenerator:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])

    def process_image(self, image_path):
        image = Image.open(image_path)
        return self.transform(image)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 10)  # Assuming input image size is 224x224

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc1(x)
        return x

class MultiModalModel:
    def __init__(self, vision_model="openai/clip-vit-base-patch32", text_model="bert-base-uncased"):
        self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(vision_model, text_model)
        self.processor = VisionTextDualEncoderProcessor.from_pretrained(vision_model)

    def predict(self, image_path, text):
        inputs = self.processor(text=[text], images=image_path, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.logits

class DataEncryptor:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt(self, data):
        return self.cipher_suite.encrypt(data.encode())

    def decrypt(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data).decode()

# Example usage
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    sample_text = "Harmful content must be filtered effectively."
    embedding = generator.generate_embedding(sample_text)
    print("Embedding shape:", embedding.shape)

    processor = ImageProcessor()
    processed_image = processor.process_image("example.jpg")
    print("Processed image shape:", processed_image.shape)

    model = SimpleCNN()
    print(model)

    multimodal_model = MultiModalModel()
    logits = multimodal_model.predict("example.jpg", "Harmful content example")
    print("Logits:", logits)

    encryptor = DataEncryptor()
    sensitive_data = "Sensitive content to be encrypted."
    encrypted = encryptor.encrypt(sensitive_data)
    print("Encrypted:", encrypted)
    decrypted = encryptor.decrypt(encrypted)
    print("Decrypted:", decrypted)
