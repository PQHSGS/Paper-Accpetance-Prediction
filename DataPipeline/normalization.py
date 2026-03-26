import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Cache stopwords once at module level (avoids reloading from disk per call)
_STOP_WORDS = set(stopwords.words('english'))
_TOKENIZER = RegexpTokenizer(r'\w+')

def normalize_text(text: str, only_char: bool = False, lower: bool = False, stop_remove: bool = False) -> str:
    """
    Normalizes the input text by filtering out non-ASCII, lowercasing, and removing stopwords.
    
    Args:
        text (str): Input text to normalize.
        only_char (bool): If True, retains only alphanumeric characters.
        lower (bool): If True, converts the text to lowercase.
        stop_remove (bool): If True, removes English stopwords.
        
    Returns:
        str: Normalized text string.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    if lower:
        text = text.lower()
    
    if only_char:
        tokens = _TOKENIZER.tokenize(text)
    else:
        tokens = text.split()
        
    if stop_remove:
        tokens = [w for w in tokens if w not in _STOP_WORDS]
        
    # Remove single-character tokens
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)
