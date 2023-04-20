import time

import gradio as gr
from gradio.components import Button, Dropdown, File, Textbox, Video
from transformers import TranslationPipeline, pipeline

from backend import utils
from backend.HelsinkiTranslator import HelsinkyTranslator
from backend.SoundExtractor import SoundExtractor
from backend.SoundTranscriptor import SoundTranscriptor
from backend.PdfExtractor import PdfExtractor

# See
# https://huggingface.co/Helsinki-NLP
# To see the possible translations.

inputSelectedModel: Dropdown = None
inputVideo: Video = None

outputAudioFile: File = None
outputTextFile: File = None
outputTranslatedFile: File = None
outputTranscriptedText: Textbox = None
outputTranslatedText: Textbox = None
outputDetectedLanguageText: Textbox = None
outputSummarizedText: Textbox = None
outputSelectedModel = None

submitButton: Button = None
translationPipeline: TranslationPipeline = None
soundExtractor: SoundExtractor = None
pdfExtractor: PdfExtractor = None
soundTranscriptor: SoundTranscriptor = None

__MAX_NUMBER_OF_TOKENS__ = 512


def performSoundExtraction(videoFilePath: str) -> str:
    """_summary_
    Returns:
        str: the mp3 file path
    """
    soundExtractor = SoundExtractor(
        videoFilePath, f"audio_out_{time.time()}.mp3")
    return soundExtractor.extractAudio()


def performSoundTranscription(pathAudioFile: str) -> tuple[str, str, str]:
    """_summary_

    Returns:
        tuple[str, str, str]: 0 => outputFileName, 1 => transcription, 2 => detectedLanguage
    """
    soundTranscriptor = SoundTranscriptor(
        pathAudioFile, f"transcription_out_{time.time()}.txt")
    return soundTranscriptor.transcribe()


def performTranslation(article: str, sourceLanguage: str, targetLanguage: str) -> tuple[str, str]:
    """_summary_

    Args:
        article (str): _description_

    Returns:
        tuple[str, str]:  0 => translation, 1 => model name
    """
    helsinkyTranslator = HelsinkyTranslator(
        article, sourceLanguage, targetLanguage)
    results = helsinkyTranslator.translate()
    return results[0], results[1]


def fetchLanguageCode(index: int) -> str:
    keys = list(utils.EUROPEAN_LANGUAGES.keys())
    result = keys[index]
    return result


def performSummarization(article: str, sourceLanguage: str, targetLanguage: str) -> str:
    """_summary_

    Args:
        article (str): _description_

    Returns:
        str: _description_
    """

    # Perform the translation to english
    result = ""
    if (sourceLanguage != "en"):
        translation = performTranslation(article, sourceLanguage, "en")[0]
    else:
        translation = article
    segments = utils.split_in_segments(translation)
    summarizer = pipeline(
        "summarization", model="philschmid/bart-large-cnn-samsum")
    print("Summarizing")
    print("===========")
    for segment in segments:
        print(segment)
        print("================================================================================")
        result = result + summarizer(segment)[0]['summary_text']

    # Perform the translation to the target language
    if (targetLanguage != "en"):
        tmpResult = ''
        segments = utils.split_in_segments(
            result, language=utils.EUROPEAN_LANGUAGES[targetLanguage].lower())
        for segment in segments:
            tmpResult = tmpResult + \
                performTranslation(segment, "en", targetLanguage)[0]
        result = tmpResult
    return result


def writeTranslationFile(sourceLanguage: str, targetLanguage: str, article: str) -> str:
    translationFileName = f"translation_out_{sourceLanguage}_{targetLanguage}_{time.time()}.txt"
    utils.writeFile(translationFileName, article)
    print(f"Translation file written: {translationFileName}")
    return translationFileName


def performPdfExtraction(pdfFilePath: str) -> str:
    pdfExtractor = PdfExtractor(
        pdfFilePath, f"transcription_out_{time.time()}.txt")
    return pdfExtractor.extractText()


def mainFunction(selectedModel, inputVideo, inputPdf):
    targetLanguage = fetchLanguageCode(selectedModel)
    outputAudioFilePath = "None"
    transcription = ["None", "None", "None"]
    translation = ["None", "None"]
    summarization  = "None"
    translationFileName = "None"

    if (targetLanguage):
        if (inputVideo):
            outputAudioFilePath = performSoundExtraction(inputVideo)
            transcription = performSoundTranscription(outputAudioFilePath)
        elif (inputPdf):
            transcription = performPdfExtraction(inputPdf.name)
        translation = performTranslation(
            transcription[1], transcription[2], targetLanguage)
        # summarization = performSummarization(
        #     article=transcription[1], sourceLanguage=transcription[2], targetLanguage=targetLanguage)
        summarization = "not yet"
        translationFileName = writeTranslationFile(
            transcription[2], targetLanguage, translation[0])
        return translation[1], outputAudioFilePath, transcription[0], translationFileName, transcription[1], transcription[2], translation[0], summarization


with gr.Blocks() as demo:
    # Inputs
    with gr.Row():
        with gr.Column():
            inputVideo = gr.Video(label="Video")
            inputPdf = gr.File(label="PDF")
    with gr.Row():
        inputSelectedModel = gr.Dropdown(
            type="index",
            choices=list(utils.EUROPEAN_LANGUAGES.values()), label="Select your target language")
    with gr.Column():
        outputAudioFile = gr.File(label="Extracted audio file")
        outputTextFile = gr.File(label="Transcripted text file")
        outputTranslatedFile = gr.File(label="Translated text file")
        outputDetectedLanguageText = gr.Textbox(
            label="Detected language", max_lines=1, lines=1)
        outputSelectedModel = gr.Textbox(label="Selected model")
    with gr.Row():
        outputTranscriptedText = gr.Textbox(
            label="Transcripted text", lines=10, max_lines=10)
        outputTranslatedText = gr.Textbox(
            label="Translated text", lines=10, max_lines=10)
    with gr.Row():
        outputSummarizedText = gr.Textbox(
            label="Summarized text", lines=3, max_lines=3)
    with gr.Row():
        submitButton = gr.Button()
        submitButton.click(fn=mainFunction, inputs=[
            inputSelectedModel, inputVideo, inputPdf], outputs=[outputSelectedModel,
                                                                outputAudioFile,
                                                                outputTextFile,
                                                                outputTranslatedFile,
                                                                outputTranscriptedText,
                                                                outputDetectedLanguageText,
                                                                outputTranslatedText,
                                                                outputSummarizedText])

demo.launch(debug=True, server_port=8080, server_name="0.0.0.0")
