from transformers import pipeline, TranslationPipeline
import time
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

    def __segmentize(this) -> str:
        segments = utils.split_in_segments(
            this.article, language=utils.EUROPEAN_LANGUAGES[this.sourceLanguage].lower())
        return segments

    def translate(this) -> tuple[str, str]:
        """_summary_

        Args:
            this (_type_): _description_

        Returns:
            tuple[str, str]: 0 => translation, 1 => model
        """
        translation = ''
        model = this.getModelName()
        try:
            segments = this.__segmentize()
            translator = pipeline(task="translation", model=model)
            counter = 0
            for segment in segments:
                tmp = translator(segment)[0]
                if tmp:
                    translatedSentence = tmp['translation_text']
                    translation = translation + translatedSentence
                    print(
                        "================================================================================")
                    print(
                        f"Translated sentence {counter}: {translatedSentence}")
                    counter += 1
        except:
            translation = f"Error: model {model} not available"
        return translation, model
