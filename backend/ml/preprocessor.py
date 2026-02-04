
"""Text preprocessing for mental health analysis."""

import re
from typing import List


class TextPreprocessor:
    """Cleans and normalizes text for ML analysis."""

    CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "can't": "cannot", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "it's": "it is", "that's": "that is", "there's": "there is",
        "what's": "what is", "let's": "let us", "who's": "who is",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "they're": "they are", "they've": "they have", "they'll": "they will",
        "we're": "we are", "we've": "we have", "we'll": "we will",
    }

    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        for contraction, expansion in self.CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r"[^a-z0-9\s.,!?'-]", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        text = self.clean(text)
        return re.findall(r'\b\w+\b', text)


