from langdetect import detect
import re

def validate_english(text: str) -> bool:
    """
    Validate whether the given text is written in English.

    This function is designed for longer prompts (sentences or paragraphs).
    It applies the `langdetect` library, but with graceful fallbacks to avoid
    false negatives. Very short inputs (≤ 1 word) are automatically accepted,
    because language detection is unreliable in such cases.

    Args:
        text (str):
            The text to validate.

    Returns:
        bool:
            True if the text is likely English or cannot be reliably detected.
            False only if the detector explicitly determines a non-English language.

    Notes:
        - `langdetect` can throw exceptions for short or ambiguous text; in such
          cases, the function returns True instead of blocking user input.
        - Designed for validating system/user prompts rather than class labels.
    """
    text = text.strip()
    if len(text.split()) <= 1:
        return True
    
    try:
        return detect(text) == "en"
    except:
        return True

def validate_label(label: str) -> bool:
    """
    Validate a class label used in the RAG-Vision frontend.

    This validator applies strict requirements for labels because they are
    used to name support folders, appear in UI elements, and influence retrieval logic.

    Rules enforced:
        - Label must not be empty.
        - Must contain only lowercase ASCII letters or hyphens.
        - Must be at most 1–2 words (hyphenated is acceptable).
        - No language detection is applied because short text is unreliable.

    Args:
        label (str):
            The class label string provided by the user.

    Returns:
        bool:
            True if the label meets all constraints, False otherwise.

    Examples:
        Valid:
            - "dirty"
            - "clean"
            - "high-contrast"
        Invalid:
            - "" (empty)
            - "clean1" (digits)
            - "very dirty" (more than 2 words)
            - "café" (non-ASCII)
    """
    if not label:
        return False

    label = label.strip().lower()

    if not re.fullmatch(r"[a-z\-]+", label):
        return False

    if len(label.split()) > 2:
        return False

    return True