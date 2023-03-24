import nltk
from nltk.tokenize import sent_tokenize

__MAX_TOKENS__ = 256

EUROPEAN_LANGUAGES = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish"
}


def split_in_segments(text="", language="english") -> list[str]:
    # nltk.download()
    tokens = 0
    mystring = list()
    segments = []
    try:
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
    except :
        print(f"!!! NO PRETRAINED MODEL FOR {language} !!!")
        segments = text.split(". ") 
    return (segments)

def writeFile(fileName: str, article: str):
    with open(fileName, "w") as file:
        file.write(article)
