import nltk
from nltk.tokenize import sent_tokenize

__MAX_TOKENS__ = 256

def split_in_segments(text="",language="English") -> list[str]:
    #nltk.download()
    tokens = 0
    mystring = list()
    segments = []
    sentences = sent_tokenize(text=text, language=language)
    for sentence in sentences:
        numberOfTokens = len(sentence.split())
        tokens += numberOfTokens
        mystring.append(str(sentence).strip())
        if tokens > __MAX_TOKENS__:
            segments.append(" ".join(mystring))
            mystring = []
            tokens = 0
    if mystring:
        segments.append(" ".join(mystring))
    return (segments)