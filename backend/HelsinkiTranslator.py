import time
import torch

from transformers import (AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSeq2SeqLM, AutoModel)

from backend import utils


class HelsinkyTranslator:

    def __init__(this, article: str, sourceLanguage: str, targetLanguage: str):
        """
            article: str, sourceLanguage: str, targetLanguage: str
        """
        this.article = article
        this.sourceLanguage = sourceLanguage
        this.targetLanguage = targetLanguage

    def getModelName(this) -> str:
        return f"Helsinki-NLP/opus-mt-{this.sourceLanguage}-{this.targetLanguage}"

    def setArticle(this, article):
        this.article = article

    def setSourceLanguage(this, sourceLanguage):
        this.sourceLanguage = sourceLanguage

    def setTargetLanguage(this, targetLanguage):
        this.targetLanguage = targetLanguage

    def __divideForMaxToken(this) -> list[PreTrainedTokenizerBase]:

        return this.article.splitlines()

    def segmentize(this, modelName: str, text: str, maxTokens: int) -> list[str]:
        results: list[str] = []
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        basicTokens: list[str] = tokenizer.tokenize(text)
        numberOfTokens = len(basicTokens)
        r = range(numberOfTokens)
        tmp = ''
        for i in r:
            if (i % maxTokens != 0):
                idxResult = i // maxTokens
                results[idxResult] = results[idxResult] + basicTokens[i]
            else:
                tmp = basicTokens[i]
                results.append(tmp)
        resultsSize = len(results)
        for i in range(resultsSize):
            results[i] = results[i].replace('▁', ' ')
        return results

    def translate(this) -> tuple[str, str]:
        """_summary_

        Args:
            this (_type_): _description_

        Returns:
            tuple[str, str]: 0 => translation, 1 => model
        """
        counter = 0
        translation = ''
        modelName = this.getModelName()
        maxTokens = 400
        segments = this.segmentize(modelName,this.article, maxTokens)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
        model.to(torch.device(device))
        for segment in segments:
            tokenized_text = tokenizer.prepare_seq2seq_batch([segment], return_tensors='pt')
            tokenized_text.to(torch.device(device))
            # Perform translation and decode the output
            tmp = model.generate(**tokenized_text)
            translated_text = tokenizer.batch_decode(tmp, skip_special_tokens=True)[0]
            translation = translation + translated_text
            counter = counter + 1
            print(translated_text)
            print(f" {counter} of {len(segments)} sentences translated")
        return translation, model
