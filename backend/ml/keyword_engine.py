
"""Clinical keyword detection based on PHQ-9 and GAD-7."""

from dataclasses import dataclass
from typing import List, Dict
import re


@dataclass
class ClinicalFlag:
    indicator_type: str
    matched_keywords: List[str]
    severity: str
    source: str
    description: str


class KeywordEngine:
    def __init__(self):
        self.phq9_keywords = {
            "anhedonia": {
                "keywords": ["no interest", "don't care", "nothing matters", "pointless",
                             "no pleasure", "can't enjoy", "don't enjoy", "numb", "empty",
                             "meaningless", "no motivation", "lost interest"],
                "description": "Little interest or pleasure in activities",
                "severity_boost": 0
            },
            "depression": {
                "keywords": ["depressed", "hopeless", "sad", "down", "miserable",
                             "unhappy", "despair", "desperate", "worthless", "helpless",
                             "crying", "darkness", "no hope", "give up"],
                "description": "Feeling down, depressed, or hopeless",
                "severity_boost": 1
            },
            "sleep_disturbance": {
                "keywords": ["can't sleep", "insomnia", "sleep too much", "oversleep",
                             "nightmares", "trouble sleeping", "exhausted", "no sleep"],
                "description": "Sleep problems",
                "severity_boost": 0
            },
            "fatigue": {
                "keywords": ["tired", "exhausted", "no energy", "fatigue", "drained",
                             "worn out", "burnt out", "burnout", "sluggish", "lethargic"],
                "description": "Feeling tired or having little energy",
                "severity_boost": 0
            },
            "low_self_esteem": {
                "keywords": ["failure", "hate myself", "loser", "stupid", "worthless",
                             "useless", "burden", "pathetic", "disappointed", "my fault"],
                "description": "Feeling bad about yourself",
                "severity_boost": 1
            },
            "self_harm": {
                "keywords": ["kill myself", "suicide", "suicidal", "end it all",
                             "end my life", "don't want to live", "want to die",
                             "better off dead", "hurt myself", "self harm", "cutting",
                             "no reason to live", "overdose"],
                "description": "Thoughts of self-harm or suicide",
                "severity_boost": 3
            }
        }

        self.gad7_keywords = {
            "anxiety": {
                "keywords": ["anxious", "nervous", "worried", "panic", "panicking",
                             "anxiety", "stressed", "tense", "on edge", "uneasy", "scared"],
                "description": "Feeling nervous, anxious, or on edge",
                "severity_boost": 0
            },
            "uncontrollable_worry": {
                "keywords": ["can't stop worrying", "constant worry", "overthinking",
                             "racing thoughts", "mind won't stop", "spiraling"],
                "description": "Not being able to stop or control worrying",
                "severity_boost": 1
            },
            "irritability": {
                "keywords": ["irritable", "annoyed", "angry", "frustrated", "snapping",
                             "short temper", "rage", "furious"],
                "description": "Becoming easily annoyed or irritable",
                "severity_boost": 0
            },
            "fear": {
                "keywords": ["afraid", "scared", "fear", "terrified", "impending doom",
                             "going to die", "losing control"],
                "description": "Feeling afraid",
                "severity_boost": 1
            }
        }

    def _calculate_severity(self, match_count: int, severity_boost: int) -> str:
        score = match_count + severity_boost
        if score >= 3:
            return "High"
        elif score >= 1:
            return "Medium"
        return "Low"

    def _find_matches(self, text: str, keywords: List[str]) -> List[str]:
        text_lower = text.lower()
        matches = []
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(keyword)
        return matches

    def analyze(self, text: str) -> Dict:
        flags = []
        phq9_categories_matched = 0
        gad7_categories_matched = 0
        risk_indicators = []

        for indicator_type, config in self.phq9_keywords.items():
            matches = self._find_matches(text, config["keywords"])
            if matches:
                phq9_categories_matched += 1
                severity = self._calculate_severity(len(matches), config["severity_boost"])
                flags.append(ClinicalFlag(
                    indicator_type=indicator_type,
                    matched_keywords=matches,
                    severity=severity,
                    source="PHQ-9",
                    description=config["description"]
                ))
                if indicator_type == "self_harm":
                    risk_indicators.append({"type": "self_harm", "severity": "Critical", "matches": matches})

        for indicator_type, config in self.gad7_keywords.items():
            matches = self._find_matches(text, config["keywords"])
            if matches:
                gad7_categories_matched += 1
                severity = self._calculate_severity(len(matches), config["severity_boost"])
                flags.append(ClinicalFlag(
                    indicator_type=indicator_type,
                    matched_keywords=matches,
                    severity=severity,
                    source="GAD-7",
                    description=config["description"]
                ))

        return {
            "flags": flags,
            "phq9_score": min(phq9_categories_matched * 2, 27),
            "gad7_score": min(gad7_categories_matched * 2, 21),
            "phq9_categories": phq9_categories_matched,
            "gad7_categories": gad7_categories_matched,
            "risk_indicators": risk_indicators,
            "has_critical_risk": len(risk_indicators) > 0
        }

    def get_summary(self, analysis: Dict) -> str:
        flags = analysis["flags"]
        if not flags:
            return "No significant clinical indicators detected."
        parts = []
        if analysis["has_critical_risk"]:
            parts.append("CRITICAL: Self-harm indicators detected")
        phq9_flags = [f for f in flags if f.source == "PHQ-9"]
        gad7_flags = [f for f in flags if f.source == "GAD-7"]
        if phq9_flags:
            parts.append(f"Depression indicators: {len(phq9_flags)}")
        if gad7_flags:
            parts.append(f"Anxiety indicators: {len(gad7_flags)}")
        return " | ".join(parts)


