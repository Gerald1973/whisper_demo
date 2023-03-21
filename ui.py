import gradio as gr
import whisper
from transformers import pipeline, TranslationPipeline
from gradio.components import Button, Dropdown, Video, File, Textbox
from backend import utils
from backend.SoundExtractor import SoundExtractor
from backend.SoundTranscriptor import SoundTranscriptor
from backend.HelsinkiTranslator import HelsinkyTranslator
import time


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
outputSelectedModel = None

submitButton: Button = None
translationPipeline: TranslationPipeline = None
soundExtractor: SoundExtractor = None
soundTranscriptor: SoundTranscriptor = None


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


def performTranslation(article: str, sourceLanguage: str, targetLanguage: str) -> tuple[str, str, str]:
    """_summary_

    Args:
        article (str): _description_

    Returns:
        tuple[str, str]: 0 => outputFileName, 1 => translation, 2 => model
    """
    outputFileName = f"translation_out_{sourceLanguage}_{targetLanguage}_{time.time()}.txt"
    helsinkyTranslator = HelsinkyTranslator(
        article, sourceLanguage, targetLanguage, outputFileName)
    return helsinkyTranslator.translate()


def fetchLanguageCode(index: int) -> str:
    keys = list(utils.EUROPEAN_LANGUAGES.keys())
    result = keys[index]
    return result


def mainFunction(selectedModel, inputVideo):
    targetLanguage = fetchLanguageCode(selectedModel)
    outputAudioFilePath = performSoundExtraction(inputVideo)
    transcription = performSoundTranscription(outputAudioFilePath)
    translation = performTranslation(transcription[1], transcription[2], targetLanguage)       
    return translation[2], outputAudioFilePath, transcription[0], translation[0], transcription[1], transcription[2], translation[1]


with gr.Blocks() as demo:
    # Inputs
    with gr.Row():
        inputVideo = gr.Video(label="Video")
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
            label="Transcripted text", lines=10, max_lines=20)
        outputTranslatedText = gr.Textbox(
            label="Translated text", lines=10, max_lines=20)
    with gr.Row():
        submitButton = gr.Button()
        submitButton.click(fn=mainFunction, inputs=[
            inputSelectedModel, inputVideo], outputs=[outputSelectedModel,
                                                      outputAudioFile,
                                                      outputTextFile,
                                                      outputTranslatedFile,
                                                      outputTranscriptedText,
                                                      outputDetectedLanguageText,
                                                      outputTranslatedText])

demo.launch(debug=True,server_port=8080,server_name="0.0.0.0")
