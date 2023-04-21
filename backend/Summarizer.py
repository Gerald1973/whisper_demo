from transformers import pipeline
from backend import utils
import torch


class Summarizer:

    __MAX_NUMBER_OF_TOKENS__ = 515

    def __init__(this, article: str):
        this.article = article
        this.device = 0 if torch.cuda.is_available() else -1
        this.modelName = "facebook/bart-large-cnn"
        #this.modelName = "sshleifer/distilbart-cnn-12-6"

    def setArticle(this, article):
        this.article = article

    def summarize(this) -> str:
        sentences = utils.segmentize(modelName=this.modelName,text=this.article,maxTokens=this.__MAX_NUMBER_OF_TOKENS__-3)
        summarizer = pipeline("summarization", model=this.modelName, device=this.device)
        counter = 0
        result = ""
        for sentence in sentences:
            counter = counter + 1
            print(f"Summarizing segment {counter} of {len(sentences)}")
            try:
                tmp = summarizer(
                    sentence, max_length=this.__MAX_NUMBER_OF_TOKENS__, min_length=30, do_sample=False)
                print(tmp[0]['summary_text'])
                print(
                    "================================================================================")
                result = result + tmp[0]['summary_text']
            except IndexError:
                print("Error")
        return result
