"""Text preprocessing module for MindGuard."""

import re
import logging
from typing import List, Set

import pandas as pd
import spacy

from src.utils import load_config

logger = logging.getLogger("mindguard")

# Load config for emotional words to preserve
try:
    CONFIG = load_config()
    KEEP_EMOTIONAL_WORDS = set(CONFIG["preprocessing"]["keep_emotional_words"])
except Exception:
    KEEP_EMOTIONAL_WORDS = {
        "alone", "hopeless", "tired", "worthless", "empty", "numb",
        "anxious", "scared", "overwhelmed", "depressed", "sad",
        "afraid", "worried", "exhausted", "helpless"
    }

# Compile regex patterns once
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
USERNAME_PATTERN = re.compile(r'\bu/\w+|\[deleted\]|\[removed\]')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?\'"-]')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, mentions, and special characters.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = URL_PATTERN.sub(' ', text)

    # Remove mentions (@username)
    text = MENTION_PATTERN.sub(' ', text)

    # Remove hashtags (keep the word, remove #)
    text = HASHTAG_PATTERN.sub(lambda m: m.group()[1:], text)

    # Remove Reddit-specific patterns
    text = USERNAME_PATTERN.sub(' ', text)

    # Remove special characters (keep basic punctuation)
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()


def anonymize_text(text: str) -> str:
    """
    Anonymize text by replacing PII with placeholders.

    Args:
        text: Input text.

    Returns:
        Anonymized text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace emails
    text = EMAIL_PATTERN.sub('[EMAIL]', text)

    # Replace phone numbers
    text = PHONE_PATTERN.sub('[PHONE]', text)

    # Replace Reddit usernames
    text = USERNAME_PATTERN.sub('[USER]', text)

    # Replace @mentions
    text = MENTION_PATTERN.sub('[USER]', text)

    return text


def load_spacy_model():
    """
    Load spaCy model with error handling.

    Returns:
        spaCy Language model.
    """
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        return nlp
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please install it.")
        raise


def lemmatize_text(
        text: str,
        nlp,
        keep_words: Set[str] = None
) -> str:
    """
    Lemmatize text while preserving emotionally significant words.

    Args:
        text: Input text.
        nlp: spaCy Language model.
        keep_words: Set of words to preserve (not lemmatize).

    Returns:
        Lemmatized text.
    """
    if not text or not isinstance(text, str):
        return ""

    if keep_words is None:
        keep_words = KEEP_EMOTIONAL_WORDS

    doc = nlp(text)

    tokens = []
    for token in doc:
        # Skip stopwords unless they're emotional words
        if token.is_stop and token.text.lower() not in keep_words:
            continue

        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue

        # Keep emotional words as-is, lemmatize others
        if token.text.lower() in keep_words:
            tokens.append(token.text.lower())
        else:
            tokens.append(token.lemma_.lower())

    return ' '.join(tokens)


def preprocess_pipeline(
        text: str,
        nlp=None,
        anonymize: bool = True,
        lemmatize: bool = True
) -> str:
    """
    Full preprocessing pipeline.

    Args:
        text: Raw input text.
        nlp: spaCy model (loaded if not provided).
        anonymize: Whether to anonymize PII.
        lemmatize: Whether to lemmatize text.

    Returns:
        Preprocessed text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Anonymize (before cleaning to catch patterns)
    if anonymize:
        text = anonymize_text(text)

    # Step 2: Clean
    text = clean_text(text)

    # Step 3: Lemmatize (optional)
    if lemmatize:
        if nlp is None:
            nlp = load_spacy_model()
        text = lemmatize_text(text, nlp)

    return text


def preprocess_dataset(
        df: pd.DataFrame,
        text_column: str,
        output_column: str = "processed_text",
        anonymize: bool = True,
        lemmatize: bool = True,
        show_progress: bool = True
) -> pd.DataFrame:
    """
    Preprocess an entire dataset.

    Args:
        df: Input DataFrame.
        text_column: Name of column containing text.
        output_column: Name for processed text column.
        anonymize: Whether to anonymize PII.
        lemmatize: Whether to lemmatize.
        show_progress: Whether to show progress bar.

    Returns:
        DataFrame with processed text column added.
    """
    from tqdm import tqdm

    df = df.copy()

    # Load spaCy model once
    nlp = load_spacy_model() if lemmatize else None

    # Process texts
    if show_progress:
        tqdm.pandas(desc="Preprocessing")
        df[output_column] = df[text_column].progress_apply(
            lambda x: preprocess_pipeline(x, nlp, anonymize, lemmatize)
        )
    else:
        df[output_column] = df[text_column].apply(
            lambda x: preprocess_pipeline(x, nlp, anonymize, lemmatize)
        )

    # Remove empty rows
    original_len = len(df)
    df = df[df[output_column].str.len() > 0]
    removed = original_len - len(df)

    if removed > 0:
        logger.info(f"Removed {removed} empty rows after preprocessing")

    return df


def validate_text(text: str, min_length: int = 10, max_length: int = 512) -> bool:
    """
    Validate text meets length requirements.

    Args:
        text: Text to validate.
        min_length: Minimum character length.
        max_length: Maximum character length.

    Returns:
        True if valid.
    """
    if not text or not isinstance(text, str):
        return False

    length = len(text.strip())
    return min_length <= length <= max_length


# Quick test
if __name__ == "__main__":
    test_texts = [
        "I've been feeling so hopeless lately. Nothing seems to matter anymore. @friend check this out https://example.com",
        "Can't sleep at night, always tired during the day. My email is test@email.com",
        "I feel alone and worthless. Nobody understands what I'm going through u/someone",
    ]

    print("Testing preprocessing pipeline:\n")
    nlp = load_spacy_model()

    for i, text in enumerate(test_texts, 1):
        print(f"--- Text {i} ---")
        print(f"Original: {text}")
        processed = preprocess_pipeline(text, nlp)
        print(f"Processed: {processed}")
        print()