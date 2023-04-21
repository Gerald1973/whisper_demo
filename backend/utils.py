from transformers import (AutoTokenizer)
import nltk
from nltk.tokenize import sent_tokenize

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

def sentencize(text: str, maxTokens: int, language="english") -> list[str]:
    # nltk.download()
    tokens = 0
    results = []
    sentences = sent_tokenize(text=text, language=language)
    tmpString = ""
    for sentence in sentences:
        numberOfTokens = len(sentence.split())
        tokens += numberOfTokens
        if tokens < maxTokens :
            tmpSentence = str(sentence).strip()
            tmpString = tmpString + " " + sentence
        else :
            results.append(tmpString)
            tmpSentence = str(sentence).strip()
            tmpString = tmpSentence
            tokens = numberOfTokens
    return results
                
def segmentize(modelName: str, text: str, maxTokens: int) -> list[str]:
        results: list[str] = []
        listOfTokens: list[list[str]] = []
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        basicTokens: list[str] = tokenizer.tokenize(text)
        numberOfTokens = len(basicTokens)
        r = range(numberOfTokens)
        tmp = ''
        for i in r:
            idxResult = i // maxTokens
            if (i % maxTokens != 0):
                listOfTokens[idxResult].append(basicTokens[i])
            else:
                tmp = basicTokens[i]
                listOfTokens.append([])
                listOfTokens[idxResult].append(tmp)
        numberOfListOfTokens = len(listOfTokens)
        for i in range(numberOfListOfTokens):
             decodedToken: str = tokenizer.convert_tokens_to_string(listOfTokens[i]);
             results.append(decodedToken)
        return results

def writeFile(fileName: str, article: str):
    with open(fileName, "w") as file:
        file.write(article)
