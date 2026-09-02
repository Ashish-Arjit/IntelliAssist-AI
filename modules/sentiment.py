"""Sentiment Analysis module for IntelliAssist AI.

Analyzes the sentiment of document content using a Hugging Face NLP classification model
with a robust lexicon-based NLP fallback for reliable offline viva demonstration.
Classifies text into Positive, Negative, or Neutral with confidence scores.
"""

from typing import Any, Dict, List, Optional
import logging
import re

from config import DEFAULT_SENTIMENT_MODEL

logger = logging.getLogger(__name__)

# Standard Sentiment Labels
SENTIMENT_POSITIVE = "Positive"
SENTIMENT_NEGATIVE = "Negative"
SENTIMENT_NEUTRAL = "Neutral"

# Sentiment Lexicon for robust rule-based fallback
POSITIVE_WORDS = {
    "great", "excellent", "good", "positive", "successful", "outstanding", "impressive",
    "profit", "profitable", "growth", "increase", "improve", "improved", "improvement",
    "benefit", "advantage", "strength", "strong", "effective", "efficient", "superior",
    "innovative", "leader", "leading", "reliable", "high-quality", "achieved", "gain",
    "promising", "valuable", "satisfactory", "happy", "remarkable", "pleased", "opportunity",
}

NEGATIVE_WORDS = {
    "bad", "poor", "negative", "failure", "failed", "decline", "decreased", "loss",
    "deficit", "drop", "dropped", "risk", "risks", "threat", "weakness", "weak",
    "problem", "issue", "trouble", "flaw", "flawed", "vulnerable", "difficult",
    "delay", "delayed", "crisis", "damage", "damaged", "error", "fault", "disappointing",
    "worst", "concern", "critical", "costly", "penalty", "loss", "losses", "uncertain",
}

INTENSIFIERS = {"very", "extremely", "highly", "substantially", "significantly", "exceptionally"}
NEGATIONS = {"not", "never", "no", "without", "neither", "hardly", "barely"}


class SentimentAnalyzer:
    """Performs document text sentiment analysis with Hugging Face models and fallback."""

    def __init__(self, model_name: str = DEFAULT_SENTIMENT_MODEL):
        """Initialize the SentimentAnalyzer.

        Args:
            model_name: Hugging Face model repository identifier.
        """
        self.model_name = model_name
        self._pipeline = None
        self._initialization_attempted = False

    def _get_pipeline(self):
        """Lazily initialize and cache the Hugging Face sentiment pipeline."""
        if not self._initialization_attempted and self._pipeline is None:
            self._initialization_attempted = True
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    top_k=None,
                )
            except Exception as e:
                logger.warning("Could not load Hugging Face sentiment model '%s': %s", self.model_name, str(e))
                self._pipeline = None
        return self._pipeline

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        """Map raw model label strings to standardized Positive / Negative / Neutral.

        Args:
            raw_label: Model prediction label (e.g. 'LABEL_0', 'positive', 'POS').

        Returns:
            Normalized label ('Positive', 'Negative', or 'Neutral').
        """
        lbl = str(raw_label).strip().lower()
        if "pos" in lbl or lbl == "label_2":
            return SENTIMENT_POSITIVE
        elif "neg" in lbl or lbl == "label_0":
            return SENTIMENT_NEGATIVE
        elif "neu" in lbl or lbl == "label_1":
            return SENTIMENT_NEUTRAL
        return SENTIMENT_NEUTRAL

    @classmethod
    def analyze_lexicon(cls, text: str) -> Dict[str, Any]:
        """Perform rule-based sentiment scoring using word valence and negations.

        Args:
            text: Input text string.

        Returns:
            Dictionary containing normalized sentiment, confidence score, and distribution.
        """
        words = re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b", text.lower())
        if not words:
            return {
                "sentiment": SENTIMENT_NEUTRAL,
                "confidence": 0.0,
                "score": 0.0,
                "scores": {SENTIMENT_POSITIVE: 0.33, SENTIMENT_NEUTRAL: 0.34, SENTIMENT_NEGATIVE: 0.33},
                "method": "Lexicon Fallback",
                "status": "warning",
                "error": "Empty or non-alphabetic text",
            }

        pos_count = 0.0
        neg_count = 0.0

        for idx, word in enumerate(words):
            # Check for negation in preceding 2 tokens
            negated = any(
                words[max(0, idx - i)] in NEGATIONS
                for i in range(1, min(3, idx + 1))
            )

            # Check for intensifier immediately preceding
            multiplier = 1.5 if (idx > 0 and words[idx - 1] in INTENSIFIERS) else 1.0

            if word in POSITIVE_WORDS:
                if negated:
                    neg_count += 1.0 * multiplier
                else:
                    pos_count += 1.0 * multiplier
            elif word in NEGATIVE_WORDS:
                if negated:
                    pos_count += 1.0 * multiplier
                else:
                    neg_count += 1.0 * multiplier

        total_sentiment_tokens = pos_count + neg_count
        if total_sentiment_tokens == 0:
            return {
                "sentiment": SENTIMENT_NEUTRAL,
                "confidence": 0.85,
                "score": 0.85,
                "scores": {SENTIMENT_POSITIVE: 0.15, SENTIMENT_NEUTRAL: 0.70, SENTIMENT_NEGATIVE: 0.15},
                "method": "Lexicon Fallback",
                "status": "success",
                "error": None,
            }

        compound = (pos_count - neg_count) / (total_sentiment_tokens + 1.0)
        pos_prob = pos_count / (total_sentiment_tokens + 2.0)
        neg_prob = neg_count / (total_sentiment_tokens + 2.0)
        neu_prob = 2.0 / (total_sentiment_tokens + 2.0)

        # Normalize probabilities so sum is 1.0
        total_p = pos_prob + neg_prob + neu_prob
        pos_score = round(pos_prob / total_p, 4)
        neg_score = round(neg_prob / total_p, 4)
        neu_score = round(neu_prob / total_p, 4)

        if compound > 0.15:
            pred_sentiment = SENTIMENT_POSITIVE
            conf = pos_score
        elif compound < -0.15:
            pred_sentiment = SENTIMENT_NEGATIVE
            conf = neg_score
        else:
            pred_sentiment = SENTIMENT_NEUTRAL
            conf = neu_score

        return {
            "sentiment": pred_sentiment,
            "confidence": round(conf, 4),
            "score": round(conf, 4),
            "scores": {
                SENTIMENT_POSITIVE: pos_score,
                SENTIMENT_NEUTRAL: neu_score,
                SENTIMENT_NEGATIVE: neg_score,
            },
            "method": "Lexicon Fallback",
            "status": "success",
            "error": None,
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of document content.

        Args:
            text: Document text or excerpt to analyze.

        Returns:
            Dictionary with sentiment ('Positive', 'Negative', 'Neutral'), confidence score, and metrics.
        """
        cleaned = (text or "").strip()

        # 1. Validation: Empty Input
        if not cleaned:
            return {
                "sentiment": SENTIMENT_NEUTRAL,
                "confidence": 0.0,
                "score": 0.0,
                "scores": {SENTIMENT_POSITIVE: 0.0, SENTIMENT_NEUTRAL: 1.0, SENTIMENT_NEGATIVE: 0.0},
                "method": "none",
                "status": "warning",
                "error": "No text provided for sentiment analysis.",
            }

        # 2. Try Hugging Face Transformer Pipeline
        pipeline_instance = self._get_pipeline()

        if pipeline_instance is not None:
            try:
                # Truncate to maximum characters safely to avoid exceeding model's 512-token limit
                safe_snippet = cleaned[:1800]
                preds = pipeline_instance(safe_snippet)

                # Format predictions
                # When top_k=None, preds is a list of lists of dicts e.g. [[{'label': ..., 'score': ...}]]
                # or a list of dicts [{'label': ..., 'score': ...}]
                if isinstance(preds, list) and len(preds) > 0:
                    first_item = preds[0]
                    items = first_item if isinstance(first_item, list) else preds

                    score_map = {SENTIMENT_POSITIVE: 0.0, SENTIMENT_NEUTRAL: 0.0, SENTIMENT_NEGATIVE: 0.0}
                    top_label = SENTIMENT_NEUTRAL
                    top_score = 0.0

                    for p in items:
                        lbl = self.normalize_label(p.get("label", ""))
                        sc = float(p.get("score", 0.0))
                        score_map[lbl] = max(score_map[lbl], sc)
                        if sc > top_score:
                            top_score = sc
                            top_label = lbl

                    return {
                        "sentiment": top_label,
                        "confidence": round(top_score, 4),
                        "score": round(top_score, 4),
                        "scores": {
                            SENTIMENT_POSITIVE: round(score_map[SENTIMENT_POSITIVE], 4),
                            SENTIMENT_NEUTRAL: round(score_map[SENTIMENT_NEUTRAL], 4),
                            SENTIMENT_NEGATIVE: round(score_map[SENTIMENT_NEGATIVE], 4),
                        },
                        "method": "Hugging Face Transformer",
                        "status": "success",
                        "error": None,
                    }
            except Exception as hf_err:
                logger.warning("Hugging Face inference error: %s. Falling back to lexicon analysis.", str(hf_err))

        # 3. Graceful Lexicon Fallback
        return self.analyze_lexicon(cleaned)
