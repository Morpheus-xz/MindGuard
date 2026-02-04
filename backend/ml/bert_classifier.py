
"""BERT-based mental health risk classifier."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional
import numpy as np


class BertClassifier:
    LABELS = ["Low", "Medium", "High"]

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._initialized = False

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self) -> bool:
        try:
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=3
                )
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self._initialized = False
            return False

    def is_loaded(self) -> bool:
        return self._initialized

    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        if not self._initialized:
            return self._mock_predict(text)

        try:
            inputs = self.tokenizer(
                text, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            result = {
                "risk_level": self.LABELS[predicted_class],
                "confidence": confidence,
                "predicted_class": predicted_class
            }

            if return_probabilities:
                result["probabilities"] = {
                    label: float(prob)
                    for label, prob in zip(self.LABELS, probabilities)
                }
            return result
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_predict(text)

    def _mock_predict(self, text: str) -> Dict:
        text_lower = text.lower()
        high_risk = ["suicide", "kill myself", "end my life", "want to die", "self harm"]
        medium_risk = ["depressed", "hopeless", "anxious", "panic", "desperate", "overwhelmed"]

        for kw in high_risk:
            if kw in text_lower:
                return {"risk_level": "High", "confidence": 0.85, "predicted_class": 2,
                        "probabilities": {"Low": 0.05, "Medium": 0.10, "High": 0.85}, "is_mock": True}

        medium_count = sum(1 for kw in medium_risk if kw in text_lower)
        if medium_count >= 2:
            return {"risk_level": "Medium", "confidence": 0.75, "predicted_class": 1,
                    "probabilities": {"Low": 0.15, "Medium": 0.75, "High": 0.10}, "is_mock": True}
        elif medium_count == 1:
            return {"risk_level": "Medium", "confidence": 0.60, "predicted_class": 1,
                    "probabilities": {"Low": 0.30, "Medium": 0.60, "High": 0.10}, "is_mock": True}

        return {"risk_level": "Low", "confidence": 0.70, "predicted_class": 0,
                "probabilities": {"Low": 0.70, "Medium": 0.25, "High": 0.05}, "is_mock": True}

    def get_model_info(self) -> Dict:
        return {"model_name": self.model_name, "model_path": self.model_path,
                "device": self.device, "is_loaded": self._initialized, "labels": self.LABELS}

