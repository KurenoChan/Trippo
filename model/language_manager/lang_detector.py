import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def detect_language(text):
    """
    Detects the language of the given text.

    Args:
        text (str): The text to detect the language for.

    Returns:
        str: The detected language code (e.g., 'en', 'fr', 'zh-cn') or 'en' (default) if detection fails.
    """
    if not text or len(text.strip()) < 5:  # Handle empty or very short text
        return "en"

    try:
        detected_language = detect(text)

        # Handle specific misclassifications
        if detected_language == "ko" and any(
            "\u4e00" <= char <= "\u9fff" for char in text
        ):
            return "zh-cn"  # Correct misclassification of Chinese as Korean

        return detected_language
    except LangDetectException:
        print(f"Language detection failed for text: {text}")  # Debugging log
        return "en"  # Default to English if detection fails