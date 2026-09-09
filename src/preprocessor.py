"""
preprocessor.py
Text cleaning, normalization, and tokenization utilities.
Supports strict ablation modes:
  - Preprocessing Enabled: lowercasing, punctuation stripping, stopword removal.
  - Preprocessing Disabled: raw text passed through unchanged.
"""

import re
import string

# Standard English stopwords list (scikit-learn style minimal list to avoid NLTK download dependencies)
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
    'but', 'by', 'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 
    'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 
    'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    's', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 
    'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 
    'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Standard preprocessing pipeline:
    1. Lowercase text
    2. Remove URLs and digits/special tokens
    3. Remove punctuation
    4. Tokenize and filter standard English stopwords
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Replace URLs and phone/numeric strings with generic tokens or spaces
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)

    # 3. Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 4. Tokenization and stopword removal
    tokens = text.split()
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in ENGLISH_STOPWORDS and len(tok) > 1]

    return " ".join(tokens)


def preprocess_corpus(texts, enabled: bool = True):
    """
    Applies text preprocessing across an iterable of texts.
    If enabled is False, returns the raw texts unmodified (for ablation study).
    """
    if not enabled:
        return [str(t) for t in texts]
    return [clean_text(str(t), remove_stopwords=True) for t in texts]


if __name__ == "__main__":
    sample = "WINNER!! As a valued network customer you have been selected 2 receive a £900 prize reward! Call 09061701461 to claim http://claim.com"
    print("Original:", sample)
    print("Preprocessed (Enabled):", clean_text(sample, remove_stopwords=True))
    print("Preprocessed (Disabled):", sample)
