"""SHAP explainer for model interpretability."""

import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path


class ShapExplainer:
    """Generate SHAP explanations for BERT predictions."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.explainer = None
        self._initialized = False
        self.class_names = ["Low", "Medium", "High"]
    
    def load(self) -> bool:
        """Load model and create SHAP explainer."""
        try:
            import shap
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            if not self.model_path:
                print("No model path provided for SHAP")
                return False
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            self.explainer = shap.Explainer(
                self._predict_proba,
                self.tokenizer,
                output_names=self.class_names
            )
            
            self._initialized = True
            print("✓ SHAP explainer loaded")
            return True
            
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return False
        except Exception as e:
            print(f"Error loading SHAP explainer: {e}")
            return False
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for SHAP."""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.numpy()
    
    def explain(self, text: str, top_k: int = 10) -> Dict:
        """Generate SHAP explanation for text."""
        if not self._initialized:
            return self._mock_explain(text)
        
        try:
            shap_values = self.explainer([text])
            tokens = self.tokenizer.tokenize(text)
            probs = self._predict_proba(text)[0]
            predicted_class = int(np.argmax(probs))
            
            values = shap_values.values[0]
            if len(values.shape) > 1:
                values = values[:, predicted_class]
            
            word_contributions = []
            for i, token in enumerate(tokens[:len(values)]):
                value = float(values[i]) if i < len(values) else 0.0
                word_contributions.append({
                    "word": token.replace("##", ""),
                    "contribution": round(value, 4),
                    "direction": "risk" if value > 0 else "protective"
                })
            
            word_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            
            top_risk = [w for w in word_contributions if w["contribution"] > 0][:top_k]
            top_protective = [w for w in word_contributions if w["contribution"] < 0][:top_k]
            
            return {
                "top_risk_words": top_risk,
                "top_protective_words": top_protective,
                "word_contributions": word_contributions[:20],
                "predicted_class": predicted_class,
                "predicted_label": self.class_names[predicted_class],
                "probabilities": {name: round(float(p), 4) for name, p in zip(self.class_names, probs)}
            }
            
        except Exception as e:
            print(f"SHAP error: {e}")
            return self._mock_explain(text)
    
    def _mock_explain(self, text: str) -> Dict:
        """Mock explanation when SHAP not available."""
        risk_keywords = {
            "hopeless": 0.25, "depressed": 0.22, "suicide": 0.35, "suicidal": 0.35,
            "anxious": 0.18, "tired": 0.12, "sad": 0.15, "alone": 0.14,
            "worthless": 0.28, "empty": 0.16, "crying": 0.13, "panic": 0.20,
            "die": 0.30, "hurt": 0.18, "pain": 0.14, "suffering": 0.16
        }
        
        protective_keywords = {
            "happy": -0.15, "good": -0.10, "better": -0.12, "hope": -0.14,
            "friends": -0.11, "family": -0.10, "love": -0.13, "grateful": -0.12
        }
        
        text_lower = text.lower()
        contributions = []
        
        for word in text_lower.split():
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in risk_keywords:
                contributions.append({"word": clean_word, "contribution": risk_keywords[clean_word], "direction": "risk"})
            elif clean_word in protective_keywords:
                contributions.append({"word": clean_word, "contribution": protective_keywords[clean_word], "direction": "protective"})
        
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        return {
            "top_risk_words": [w for w in contributions if w["direction"] == "risk"][:10],
            "top_protective_words": [w for w in contributions if w["direction"] == "protective"][:10],
            "word_contributions": contributions,
            "is_mock": True
        }
    
    def get_visualization_data(self, text: str) -> Dict:
        """Format explanation for frontend charts."""
        explanation = self.explain(text)
        return {
            "risk_factors": [{"word": w["word"], "score": round(w["contribution"] * 100, 1)} for w in explanation.get("top_risk_words", [])],
            "protective_factors": [{"word": w["word"], "score": round(abs(w["contribution"]) * 100, 1)} for w in explanation.get("top_protective_words", [])],
            "all_contributions": [{"word": w["word"], "score": round(w["contribution"] * 100, 1), "type": w["direction"]} for w in explanation.get("word_contributions", [])]
        }
