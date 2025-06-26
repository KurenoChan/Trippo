import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model.language_manager.lang_reservedWord as ReservedWordManager
import model.language_manager.lang_detector as LanguageDetector
import translators as ts

def translate_text(text, target_language):
    """
    Translates the given text to the target language, excluding reserved words.

    Args:
        text (str): The text to translate.
        target_language (str): The target language code (e.g., 'en', 'fr', 'es').

    Returns:
        str: The translated text.
    """
    if not text or text.strip() == "":
        return ""  # Return an empty string if input is empty

    detected_language = LanguageDetector.detect_language(text)

    if detected_language == target_language:
        return text  # No translation needed if already in target language

    try:
        # Split into words while preserving whitespace for reconstruction
        words = text.split(' ')
        translated_words = []
        
        for word in words:
            if not word.strip():  # Skip empty strings from multiple spaces
                translated_words.append(word)
                continue
                
            if ReservedWordManager.is_reserved_word(word):
                translated_words.append(word)  # Keep reserved words as is
            else:
                # Translate individual words with context
                translation = ts.translate_text(
                    word, 
                    to_language=target_language,
                    from_language=detected_language
                )
                translated_words.append(translation)
        
        # Reconstruct the text with original spacing
        return ' '.join(translated_words)
        
    except Exception as e:
        print(f"Translation error: {e}")  # Debugging log
        return text  # Return original text if translation fails