"""Main ML pipeline combining all components."""

from typing import Dict, List, Optional
from pathlib import Path

from .preprocessor import TextPreprocessor
from .keyword_engine import KeywordEngine
from .bert_classifier import BertClassifier
from .explainer import ShapExplainer
from .vector_store import VectorStore
from .lstm_predictor import TrendPredictor


class MentalHealthPipeline:
    """
    Complete mental health screening pipeline.
    
    Components:
    1. Text Preprocessor - cleans input
    2. BERT Classifier - predicts risk level
    3. Keyword Engine - detects PHQ-9/GAD-7 indicators
    4. SHAP Explainer - explains predictions
    5. Vector Store (ChromaDB) - finds similar cases
    6. LSTM Predictor - forecasts 7-day trend
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        model_path: Optional[str] = None,
        load_model: bool = False,
        enable_shap: bool = True,
        enable_chromadb: bool = True,
        enable_lstm: bool = True
    ):
        # Core components
        self.preprocessor = TextPreprocessor()
        self.keyword_engine = KeywordEngine()
        
        # Auto-detect model path
        if model_path is None:
            base_dir = Path(__file__).parent.parent.parent / "models"
            # Check both possible model names
            possible_paths = [
                base_dir / "mindguard-distilbert-final",
                base_dir / "mindguard-bert-final"
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
        
        self.model_path = model_path
        
        # BERT Classifier
        self.classifier = BertClassifier(
            model_name=model_name,
            model_path=model_path
        )
        
        # Optional components
        self.explainer = None
        self.vector_store = None
        self.trend_predictor = None
        
        self._shap_enabled = enable_shap
        self._chromadb_enabled = enable_chromadb
        self._lstm_enabled = enable_lstm
        
        if load_model:
            self.load_all()
    
    def load_all(self) -> Dict[str, bool]:
        """Load all components."""
        results = {}
        
        # Load BERT
        results['bert'] = self.classifier.load()
        
        # Load SHAP
        if self._shap_enabled and self.model_path:
            self.explainer = ShapExplainer(self.model_path)
            results['shap'] = self.explainer.load()
        
        # Initialize ChromaDB
        if self._chromadb_enabled:
            self.vector_store = VectorStore()
            results['chromadb'] = self.vector_store.initialize()
        
        # Initialize LSTM
        if self._lstm_enabled:
            lstm_path = Path(__file__).parent.parent.parent / "models" / "lstm_trend_model.pt"
            self.trend_predictor = TrendPredictor(
                str(lstm_path) if lstm_path.exists() else None
            )
            results['lstm'] = self.trend_predictor.initialize()
        
        return results
    
    def analyze(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_shap: bool = True,
        include_similar: bool = True,
        user_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Run complete analysis on input text."""
        
        # 1. Preprocess
        cleaned_text = self.preprocessor.clean(text)
        
        # 2. BERT classification
        bert_result = self.classifier.predict(text)
        
        # 3. Keyword analysis
        keyword_result = self.keyword_engine.analyze(text)
        
        # 4. Aggregate risk
        final_risk, final_confidence = self._aggregate_risk(bert_result, keyword_result)
        
        # Build clinical flags list
        clinical_flags = [
            {
                "indicator_type": flag.indicator_type,
                "matched_keywords": ", ".join(flag.matched_keywords),
                "severity": flag.severity,
                "source": flag.source,
                "description": flag.description
            }
            for flag in keyword_result["flags"]
        ]
        
        # Build base result
        result = {
            "risk_level": final_risk,
            "confidence": final_confidence,
            "probabilities": bert_result.get("probabilities", {}),
            "clinical_flags": clinical_flags,
            "phq9_score": keyword_result["phq9_score"],
            "gad7_score": keyword_result["gad7_score"],
            "has_critical_risk": keyword_result["has_critical_risk"],
            "risk_indicators": keyword_result["risk_indicators"],
            "processed_text": cleaned_text,
            "model_prediction": bert_result["risk_level"],
            "is_mock_prediction": bert_result.get("is_mock", False),
            "summary": self.keyword_engine.get_summary(keyword_result)
        }
        
        # 5. SHAP explanations
        if include_shap and self.explainer and self.explainer._initialized:
            result["shap_explanation"] = self.explainer.get_visualization_data(text)
        elif include_shap:
            mock_explainer = ShapExplainer(None)
            result["shap_explanation"] = mock_explainer._mock_explain(text)
            result["shap_explanation"]["is_mock"] = True
        
        # 6. Similar cases from ChromaDB
        if include_similar and self.vector_store and self.vector_store._initialized:
            similar = self.vector_store.find_similar(
                text, n_results=3, exclude_user_id=user_id
            )
            result["similar_cases"] = similar
        else:
            result["similar_cases"] = []
        
        # 7. Add to vector store
        if self.vector_store and self.vector_store._initialized and session_id and user_id:
            flag_types = [f["indicator_type"] for f in clinical_flags]
            self.vector_store.add_session(
                session_id=session_id,
                text=text,
                user_id=user_id,
                risk_level=final_risk,
                confidence=final_confidence,
                clinical_flags=flag_types
            )
        
        # 8. Trend prediction
        if user_history and self.trend_predictor and self.trend_predictor._initialized:
            result["trend_forecast"] = self.trend_predictor.predict(user_history)
        else:
            result["trend_forecast"] = {
                "status": "not_available",
                "message": "No history provided or LSTM not initialized"
            }
        
        return result
    
    def _aggregate_risk(self, bert_result: Dict, keyword_result: Dict) -> tuple:
        """Aggregate risk from BERT and keywords."""
        bert_risk = bert_result["risk_level"]
        bert_confidence = bert_result["confidence"]
        
        # Critical risk override
        if keyword_result["has_critical_risk"]:
            return "High", 0.95
        
        high_severity_flags = sum(1 for f in keyword_result["flags"] if f.severity == "High")
        
        risk_scores = {"Low": 1, "Medium": 2, "High": 3}
        score_to_risk = {1: "Low", 2: "Medium", 3: "High"}
        
        bert_score = risk_scores[bert_risk]
        
        keyword_adjustment = 0
        if high_severity_flags >= 2:
            keyword_adjustment = 1
        elif keyword_result["phq9_categories"] >= 4 or keyword_result["gad7_categories"] >= 4:
            keyword_adjustment = 0.5
        
        final_score = min(3, bert_score + keyword_adjustment)
        final_risk = score_to_risk[round(final_score)]
        
        if final_risk == bert_risk:
            final_confidence = min(0.95, bert_confidence + 0.1)
        else:
            final_confidence = max(0.5, bert_confidence - 0.1)
        
        return final_risk, round(final_confidence, 2)
    
    def get_trend_forecast(self, user_history: List[Dict]) -> Dict:
        """Get trend forecast for a user."""
        if not self.trend_predictor or not self.trend_predictor._initialized:
            return {"status": "not_available", "message": "LSTM not initialized"}
        return self.trend_predictor.predict(user_history)
    
    def get_status(self) -> Dict:
        """Get pipeline status."""
        return {
            "bert": {
                "loaded": self.classifier.is_loaded(),
                "model_path": self.model_path,
                "info": self.classifier.get_model_info()
            },
            "shap": {
                "enabled": self._shap_enabled,
                "loaded": self.explainer._initialized if self.explainer else False
            },
            "chromadb": {
                "enabled": self._chromadb_enabled,
                "stats": self.vector_store.get_stats() if self.vector_store else {}
            },
            "lstm": {
                "enabled": self._lstm_enabled,
                "loaded": self.trend_predictor._initialized if self.trend_predictor else False
            }
        }
