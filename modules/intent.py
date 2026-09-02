"""Intent Analysis module for IntelliAssist AI.

Classifies natural language user queries into functional intent categories:
- Question
- Summary Request
- Information Search
- Explanation Request

Uses linguistic markers, syntactic cues, and keyword pattern scoring for
fast, deterministic, and easily explainable NLP intent classification.
"""

from typing import Any, Dict, List, Optional
import re

from config import INTENT_CATEGORIES

# Intent Category Constants
INTENT_QUESTION = "Question"
INTENT_SUMMARY = "Summary Request"
INTENT_SEARCH = "Information Search"
INTENT_EXPLANATION = "Explanation Request"

INTENT_DESCRIPTIONS = {
    INTENT_QUESTION: "Inquiry seeking specific factual or direct answers from documents.",
    INTENT_SUMMARY: "Request to condense, synthesize, or outline core document content.",
    INTENT_SEARCH: "Targeted retrieval to find mentions, data points, or topics.",
    INTENT_EXPLANATION: "Conceptual inquiry seeking in-depth reasoning, mechanism, or clarification.",
}

INTENT_ICONS = {
    INTENT_QUESTION: "❓",
    INTENT_SUMMARY: "📝",
    INTENT_SEARCH: "🔍",
    INTENT_EXPLANATION: "💡",
}


class IntentAnalyzer:
    """Classifies user queries into distinct intent categories with confidence estimation."""

    # Explicit phrase markers for high confidence
    SUMMARY_PATTERNS = [
        r"^(?:please\s+)?summarize\b",
        r"\bsummarize\s+(?:this|the|all)?\b",
        r"\b(?:give\s+me\s+a\s+)?(?:quick\s+|brief\s+|executive\s+)?summary\b",
        r"\btldr\b",
        r"\brecap\b",
        r"\bkey\s+takeaways?\b",
        r"\bmain\s+points?\b",
        r"\bbullet\s+points?\b",
        r"\boverview\s+of\b",
        r"\bcondense\b",
        r"\bin\s+a\s+nutshell\b",
    ]

    EXPLANATION_PATTERNS = [
        r"^(?:please\s+)?explain\b",
        r"\bexplain\s+(?:the|how|why|what|this)?\b",
        r"\belaborate\s+on\b",
        r"\bclarify\b",
        r"\bwhy\s+(?:did|does|is|are|was|were|would|should)\b",
        r"\bhow\s+(?:does|do|can|could|is|are|did)\b",
        r"\bwalk\s+me\s+through\b",
        r"\bdescribe\s+(?:how|the\s+process|the\s+mechanism)\b",
        r"\bconcept\s+of\b",
        r"\bdeep\s+dive\b",
    ]

    SEARCH_PATTERNS = [
        r"^(?:please\s+)?find\b",
        r"\bfind\s+(?:information|details|data|sections?|mentions?)\b",
        r"\bsearch\s+(?:for|about)?\b",
        r"\blocate\b",
        r"\blookup\b",
        r"\bwhere\s+(?:can\s+i\s+find|is|are)\b",
        r"\bshow\s+me\s+(?:where|mentions?\s+of|instances?\s+of|all)\b",
        r"\blist\s+all\b",
        r"\bretrieve\b",
        r"\blook\s+for\b",
    ]

    QUESTION_PATTERNS = [
        r"^(?:what|who|when|which|where|whom|whose)\b",
        r"^(?:is|are|was|were|can|could|would|should|do|does|did|will|has|have|had)\b",
        r"\?$",
    ]

    def __init__(self):
        """Initialize the IntentAnalyzer."""
        self.categories = INTENT_CATEGORIES

    def analyze(self, query: str) -> Dict[str, Any]:
        """Classify the user's natural language query intent.

        Args:
            query: Natural language query string.

        Returns:
            Dictionary containing detected intent, confidence score, description, icon, and matched cues.
        """
        cleaned = (query or "").strip()

        # 1. Validation: Empty Query
        if not cleaned:
            return {
                "intent": INTENT_QUESTION,
                "confidence": 0.0,
                "description": "Empty query provided.",
                "icon": INTENT_ICONS[INTENT_QUESTION],
                "badge": f"{INTENT_ICONS[INTENT_QUESTION]} {INTENT_QUESTION}",
                "matched_keywords": [],
                "status": "warning",
                "error": "Query is empty.",
            }

        q_lower = cleaned.lower()

        # 2. Priority Pattern Matching
        # Priority 1: Summary Request (e.g. "Summarize this document.")
        for pat in self.SUMMARY_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                matched_kw = match.group(0).strip()
                # Boost confidence if query starts directly with summary verb
                conf = 0.96 if q_lower.startswith(matched_kw) else 0.88
                return {
                    "intent": INTENT_SUMMARY,
                    "confidence": conf,
                    "description": INTENT_DESCRIPTIONS[INTENT_SUMMARY],
                    "icon": INTENT_ICONS[INTENT_SUMMARY],
                    "badge": f"{INTENT_ICONS[INTENT_SUMMARY]} {INTENT_SUMMARY}",
                    "matched_keywords": [matched_kw],
                    "status": "success",
                    "error": None,
                }

        # Priority 2: Explanation Request (e.g. "Explain the main concept in this document.")
        for pat in self.EXPLANATION_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                matched_kw = match.group(0).strip()
                conf = 0.95 if q_lower.startswith(matched_kw) else 0.87
                return {
                    "intent": INTENT_EXPLANATION,
                    "confidence": conf,
                    "description": INTENT_DESCRIPTIONS[INTENT_EXPLANATION],
                    "icon": INTENT_ICONS[INTENT_EXPLANATION],
                    "badge": f"{INTENT_ICONS[INTENT_EXPLANATION]} {INTENT_EXPLANATION}",
                    "matched_keywords": [matched_kw],
                    "status": "success",
                    "error": None,
                }

        # Priority 3: Information Search (e.g. "Find information about machine learning.")
        for pat in self.SEARCH_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                matched_kw = match.group(0).strip()
                conf = 0.94 if q_lower.startswith(matched_kw) else 0.86
                return {
                    "intent": INTENT_SEARCH,
                    "confidence": conf,
                    "description": INTENT_DESCRIPTIONS[INTENT_SEARCH],
                    "icon": INTENT_ICONS[INTENT_SEARCH],
                    "badge": f"{INTENT_ICONS[INTENT_SEARCH]} {INTENT_SEARCH}",
                    "matched_keywords": [matched_kw],
                    "status": "success",
                    "error": None,
                }

        # Priority 4: Question (e.g. "What is this document about?")
        for pat in self.QUESTION_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                matched_kw = match.group(0).strip()
                return {
                    "intent": INTENT_QUESTION,
                    "confidence": 0.92,
                    "description": INTENT_DESCRIPTIONS[INTENT_QUESTION],
                    "icon": INTENT_ICONS[INTENT_QUESTION],
                    "badge": f"{INTENT_ICONS[INTENT_QUESTION]} {INTENT_QUESTION}",
                    "matched_keywords": [matched_kw] if matched_kw else ["question marker"],
                    "status": "success",
                    "error": None,
                }

        # 3. Fallback Classification
        # If text ends with '?' it's a question, otherwise keyword-based information search
        if cleaned.endswith("?"):
            return {
                "intent": INTENT_QUESTION,
                "confidence": 0.82,
                "description": INTENT_DESCRIPTIONS[INTENT_QUESTION],
                "icon": INTENT_ICONS[INTENT_QUESTION],
                "badge": f"{INTENT_ICONS[INTENT_QUESTION]} {INTENT_QUESTION}",
                "matched_keywords": ["?"],
                "status": "success",
                "error": None,
            }

        # Default open search intent for unstructured keywords
        return {
            "intent": INTENT_SEARCH,
            "confidence": 0.72,
            "description": INTENT_DESCRIPTIONS[INTENT_SEARCH],
            "icon": INTENT_ICONS[INTENT_SEARCH],
            "badge": f"{INTENT_ICONS[INTENT_SEARCH]} {INTENT_SEARCH}",
            "matched_keywords": ["keyword search"],
            "status": "success",
            "error": None,
        }
