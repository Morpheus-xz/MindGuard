"""Clinical keyword detection engine aligned with PHQ-9 and GAD-7."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("mindguard")


class KeywordEngine:
    """
    Detects clinical indicators based on PHQ-9 and GAD-7 frameworks.

    PHQ-9: Patient Health Questionnaire for depression
    GAD-7: Generalized Anxiety Disorder scale
    """

    def __init__(
            self,
            phq9_path: str = "src/lexicons/phq9_keywords.json",
            gad7_path: str = "src/lexicons/gad7_keywords.json"
    ):
        """
        Initialize the keyword engine.

        Args:
            phq9_path: Path to PHQ-9 keywords JSON.
            gad7_path: Path to GAD-7 keywords JSON.
        """
        self.lexicons = self._load_lexicons(phq9_path, gad7_path)
        self._compile_patterns()
        logger.info(f"KeywordEngine initialized with {len(self.lexicons)} categories")

    def _load_lexicons(
            self,
            phq9_path: str,
            gad7_path: str
    ) -> Dict[str, Dict]:
        """
        Load and merge PHQ-9 and GAD-7 lexicons.

        Args:
            phq9_path: Path to PHQ-9 keywords.
            gad7_path: Path to GAD-7 keywords.

        Returns:
            Merged lexicon dictionary.
        """
        lexicons = {}

        # Load PHQ-9
        phq9_file = Path(phq9_path)
        if phq9_file.exists():
            with open(phq9_file, "r") as f:
                phq9_data = json.load(f)
                for category, data in phq9_data.items():
                    lexicons[category] = {
                        "keywords": data["keywords"],
                        "severity_weight": data["severity_weight"],
                        "source": "PHQ-9"
                    }
        else:
            logger.warning(f"PHQ-9 lexicon not found: {phq9_path}")

        # Load GAD-7
        gad7_file = Path(gad7_path)
        if gad7_file.exists():
            with open(gad7_file, "r") as f:
                gad7_data = json.load(f)
                for category, data in gad7_data.items():
                    lexicons[category] = {
                        "keywords": data["keywords"],
                        "severity_weight": data["severity_weight"],
                        "source": "GAD-7"
                    }
        else:
            logger.warning(f"GAD-7 lexicon not found: {gad7_path}")

        return lexicons

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.patterns = {}

        for category, data in self.lexicons.items():
            # Escape special regex characters and create pattern
            patterns = []
            for keyword in data["keywords"]:
                # Escape and create word boundary pattern
                escaped = re.escape(keyword.lower())
                patterns.append(escaped)

            # Combine into single pattern with word boundaries
            if patterns:
                combined = r'\b(' + '|'.join(patterns) + r')\b'
                self.patterns[category] = re.compile(combined, re.IGNORECASE)

    def detect_indicators(self, text: str) -> List[Dict]:
        """
        Detect clinical indicators in text.

        Args:
            text: Input text to analyze.

        Returns:
            List of detected indicators with details.
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        indicators = []

        for category, pattern in self.patterns.items():
            matches = pattern.findall(text_lower)

            if matches:
                # Get unique matches
                unique_matches = list(set(matches))

                # Calculate severity based on matches and weight
                weight = self.lexicons[category]["severity_weight"]
                match_count = len(matches)
                severity = self._calculate_indicator_severity(match_count, weight)

                indicators.append({
                    "indicator_type": category,
                    "matched_keywords": ", ".join(unique_matches),
                    "match_count": match_count,
                    "severity": severity,
                    "severity_weight": weight,
                    "source": self.lexicons[category]["source"]
                })

        # Sort by severity weight (highest first)
        indicators.sort(key=lambda x: x["severity_weight"], reverse=True)

        return indicators

    def _calculate_indicator_severity(
            self,
            match_count: int,
            weight: int
    ) -> str:
        """
        Calculate severity level for an indicator.

        Args:
            match_count: Number of keyword matches.
            weight: Category weight (1-3).

        Returns:
            Severity level: Low, Medium, or High.
        """
        # Score based on matches and weight
        score = match_count * weight

        if weight == 3:  # Critical categories (e.g., suicidal_ideation)
            return "High"
        elif score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"

    def calculate_overall_severity(self, indicators: List[Dict]) -> str:
        """
        Calculate overall severity from all indicators.

        Args:
            indicators: List of detected indicators.

        Returns:
            Overall severity: Low, Medium, or High.
        """
        if not indicators:
            return "Low"

        # If any High severity indicator, return High
        if any(ind["severity"] == "High" for ind in indicators):
            return "High"

        # If any critical category detected
        critical_categories = {"suicidal_ideation"}
        if any(ind["indicator_type"] in critical_categories for ind in indicators):
            return "High"

        # Count severity levels
        medium_count = sum(1 for ind in indicators if ind["severity"] == "Medium")

        # Multiple medium indicators escalate to High
        if medium_count >= 3:
            return "High"
        elif medium_count >= 1:
            return "Medium"

        # Multiple low indicators can escalate to Medium
        if len(indicators) >= 4:
            return "Medium"

        return "Low"

    def get_dominant_flag(self, indicators: List[Dict]) -> Optional[str]:
        """
        Get the most significant indicator.

        Args:
            indicators: List of detected indicators.

        Returns:
            Most dominant indicator type or None.
        """
        if not indicators:
            return None

        # Sort by severity weight, then by match count
        sorted_indicators = sorted(
            indicators,
            key=lambda x: (x["severity_weight"], x["match_count"]),
            reverse=True
        )

        return sorted_indicators[0]["indicator_type"]

    def get_summary(self, indicators: List[Dict]) -> Dict:
        """
        Get a summary of detected indicators.

        Args:
            indicators: List of detected indicators.

        Returns:
            Summary dictionary.
        """
        if not indicators:
            return {
                "total_indicators": 0,
                "overall_severity": "Low",
                "dominant_flag": None,
                "categories": [],
                "phq9_count": 0,
                "gad7_count": 0
            }

        phq9_count = sum(1 for ind in indicators if ind["source"] == "PHQ-9")
        gad7_count = sum(1 for ind in indicators if ind["source"] == "GAD-7")

        return {
            "total_indicators": len(indicators),
            "overall_severity": self.calculate_overall_severity(indicators),
            "dominant_flag": self.get_dominant_flag(indicators),
            "categories": [ind["indicator_type"] for ind in indicators],
            "phq9_count": phq9_count,
            "gad7_count": gad7_count
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("MindGuard Clinical Keyword Engine Test")
    print("=" * 60)

    engine = KeywordEngine()

    test_texts = [
        "I feel so hopeless and worthless. Nothing matters anymore.",
        "I can't sleep at night and I'm always tired during the day.",
        "I'm constantly worried about everything. My mind won't stop racing.",
        "Sometimes I think about hurting myself. I don't want to exist.",
        "I had a good day today, feeling pretty happy!",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")

        indicators = engine.detect_indicators(text)
        summary = engine.get_summary(indicators)

        print(f"Overall Severity: {summary['overall_severity']}")
        print(f"Dominant Flag: {summary['dominant_flag']}")
        print(f"Indicators Found: {summary['total_indicators']}")

        if indicators:
            print("Details:")
            for ind in indicators:
                print(f"  - {ind['indicator_type']}: '{ind['matched_keywords']}' ({ind['severity']})")

        print()