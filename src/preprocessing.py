import re

def cleaning(text) -> str:

    """
    Clean the input text by converting it to lowercase, removing non-alphanumeric characters, and reducing consecutive spaces to single spaces.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]','', text)
    text = re.sub(r'\s+',' ', text)
    return text
