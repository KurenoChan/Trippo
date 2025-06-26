import os
import sys

# Add the project root (one level up from 'model') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import model.language_manager.lang_reservedWord as ReservedWordManager
from autocorrect import Speller

def correct_spelling(text):
    """
    Corrects spelling errors in the given text, excluding reserved words and slang.

    Args:
        text (str): The text to correct spelling for.

    Returns:
        str: The text with corrected spelling.
    """
    if not text or text.strip() == "":
        return ""  # Return an empty string if input is empty

    try:
        spell = Speller()
        words = text.split()
        corrected_words = []

        for word in words:
            word_lower = word.lower()

            if ReservedWordManager.is_reserved_word(word):  # Skip reserved words
                corrected_words.append(word)
            elif ReservedWordManager.is_slang_word(word_lower):  # Replace slang with its full form
                corrected_words.append(ReservedWordManager.SLANG_WORDS[word_lower])
            else:
                corrected_word = spell(word_lower)  # Correct spelling if it's not reserved or slang
                corrected_words.append(corrected_word if corrected_word else word)
                
        return " ".join(corrected_words)
    except Exception as e:
        print(f"Spelling correction error: {e}")  # Debugging log
        return text  # Return original text if correction fails