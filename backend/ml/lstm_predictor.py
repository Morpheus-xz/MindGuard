"""LSTM-based trend predictor for 7-day risk forecasting."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta


class LSTMModel(nn.Module):
    """LSTM network for time-series prediction."""
    
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=7, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, output_size), nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TrendPredictor:
    """Predicts 7-day risk trend using LSTM."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self._initialized = False
        self.sequence_length = 30
        self.min_sessions = 7
    
    def initialize(self) -> bool:
        """Initialize the LSTM model."""
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            
            self.model = LSTMModel().to(self.device)
            
            if self.model_path and Path(self.model_path).exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print(f"✓ LSTM model loaded")
            else:
                print("⚠ LSTM using heuristic mode (no trained model)")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"LSTM error: {e}")
            return False
    
    def _risk_to_numeric(self, risk: str) -> float:
        return {"Low": 1.0, "Medium": 2.0, "High": 3.0}.get(risk, 2.0)
    
    def _numeric_to_risk(self, val: float) -> str:
        if val < 1.5: return "Low"
        if val < 2.5: return "Medium"
        return "High"
    
    def predict(self, sessions: List[Dict]) -> Dict:
        """Predict 7-day risk trend."""
        if len(sessions) < self.min_sessions:
            return {
                "status": "insufficient_data",
                "message": f"Need {self.min_sessions} sessions, have {len(sessions)}",
                "sessions_needed": self.min_sessions - len(sessions)
            }
        
        return self._heuristic_predict(sessions)
    
    def _heuristic_predict(self, sessions: List[Dict]) -> Dict:
        """Simple heuristic prediction."""
        recent = sessions[-7:] if len(sessions) >= 7 else sessions
        scores = [self._risk_to_numeric(s.get('risk_level', 'Medium')) for s in recent]
        
        avg_score = np.mean(scores)
        mid = len(scores) // 2
        trend = np.mean(scores[mid:]) - np.mean(scores[:mid]) if mid > 0 else 0
        
        today = datetime.utcnow().date()
        forecast = []
        
        for i in range(7):
            projected = max(1.0, min(3.0, avg_score + (trend * (i + 1) * 0.1)))
            forecast.append({
                "date": (today + timedelta(days=i+1)).isoformat(),
                "day": i + 1,
                "predicted_score": round(projected, 2),
                "predicted_level": self._numeric_to_risk(projected)
            })
        
        forecast_avg = np.mean([f['predicted_score'] for f in forecast])
        
        if forecast_avg > avg_score + 0.2:
            direction, concern = "increasing", "high" if forecast_avg > 2.5 else "moderate"
        elif forecast_avg < avg_score - 0.2:
            direction, concern = "decreasing", "low"
        else:
            direction, concern = "stable", "moderate" if avg_score > 2.0 else "low"
        
        high_risk_days = sum(1 for f in forecast if f['predicted_level'] == 'High')
        
        if high_risk_days >= 5:
            rec = "Consider reaching out to a mental health professional."
        elif direction == "increasing":
            rec = "Your risk trend is increasing. Practice self-care."
        elif direction == "decreasing":
            rec = "Positive trend! Keep up your wellness practices."
        else:
            rec = "Stable trend. Continue monitoring."
        
        return {
            "status": "success",
            "forecast": forecast,
            "trend_analysis": {
                "direction": direction,
                "concern_level": concern,
                "high_risk_days": high_risk_days,
                "recommendation": rec
            },
            "model_used": "heuristic"
        }
