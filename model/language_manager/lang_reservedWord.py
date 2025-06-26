# Define a list of reserved words that should not be translated or corrected
APP_WORDS = {"Trippo", "TrippoBot", "AI"}

SLANG_WORDS = {
    "lmao": "laughing",
    "lol": "laugh out loud",
    "brb": "be right back",
    "idk": "I don't know",
    "omg": "oh my god",
    "smh": "shaking my head",
}


def is_app_word(word):
    """
    Checks if a word is an app word.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is an app word, False otherwise.
    """
    return word.lower() in (app_word.lower() for app_word in APP_WORDS)


def is_slang_word(word):
    """
    Checks if a word is a slang word.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is a slang word, False otherwise.
    """
    return word.lower() in (slang_word.lower() for slang_word in SLANG_WORDS)


def is_reserved_word(word):
    """
    Checks if a word is a reserved word (app or slang).

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is a reserved word, False otherwise.
    """
    return is_app_word(word) or is_slang_word(word)
