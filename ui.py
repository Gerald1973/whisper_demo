import gradio as gr
import whisper
from transformers import pipeline, TranslationPipeline
from gradio.components import Button, Dropdown, Video, File, Textbox
from backend import utils
from backend.SoundExtractor import SoundExtractor
from backend.SoundTranscriptor import SoundTranscriptor
import time


# See
# https://huggingface.co/Helsinki-NLP
# To see the possible translations.

inputSelectedModel: Dropdown = None
inputVideo: Video = None
outputAudioFile: File = None
outputTextFile: File = None
outputTranscriptedText: Textbox = None
outputTranslatedText: Textbox = None
outputDetectedLanguageText: Textbox = None
outSelectedModel = None
submitButton: Button = None
translationPipeline: TranslationPipeline = None
soundExtractor: SoundExtractor = None
soundTranscriptor: SoundTranscriptor = None

modelInputs = [
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-es-fr",
    "Helsinki-NLP/opus-mt-fr-es",
    "Helsinki-NLP/opus-mt-fr-bg",
    "Helsinki-NLP/opus-mt-fr-de",
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-fr-hr",
    "Helsinki-NLP/opus-mt-fr-it",
    "Helsinki-NLP/opus-mt-fr-pl",
    "Helsinki-NLP/opus-mt-fr-pt",
    "Helsinki-NLP/opus-mt-fr-ro",
    "Helsinki-NLP/opus-mt-fr-ru",
    "Helsinki-NLP/opus-mt-fr-sv",
    "Helsinki-NLP/opus-mt-fr-tr",
    "Helsinki-NLP/opus-mt-fr-uk",
    "Helsinki-NLP/opus-mt-fr-zh",
    "t5-small",
    "t5-large"]


def performSoundExtraction(path: str) -> str:
    soundExtractor = SoundExtractor()
    soundExtractor.setVideoFile(path)
    soundExtractor.setAudioFile(f"audio_out_{time.time()}.mp3")
    return soundExtractor.extractAudio()


def performSoundTranscription(pathAudioFile: str) -> tuple[str: str, str: str, str: str]:
    soundTranscriptor = SoundTranscriptor()
    soundTranscriptor.setAudioFile(pathAudioFile)
    soundTranscriptor.setTextFile(f"text_out_{time.time()}.txt")
    return soundTranscriptor.transcribe()


def mainFunction(selectedModel, inputVideo):
    print(selectedModel)
    outputFilePath = performSoundExtraction(inputVideo)
    textFileText = performSoundTranscription(outputFilePath)
    translation = 'no model selected'
    if selectedModel:
        translation = ''
        translator = pipeline(task="translation", model=selectedModel)
        if (textFileText[2] == "fr"):
            sourceLanguage = "french"
        elif (textFileText[2] == "en"):
            sourceLanguage = "english"
        elif (textFileText[2] == "es"):
            sourceLanguage = "spanish"
        elif (textFileText[2] == "de"):
            sourceLanguage = "german"
        else:
            sourceLanguage = "english"
        segments = utils.split_in_segments(
            textFileText[1], language=sourceLanguage)
        counter = 0
        for segment in segments:
            tmp = translator(segment)[0]
            if tmp:
                translatedSentence = tmp['translation_text']
                translation = translation + translatedSentence
                print(f"Translated sentence {counter}: {translatedSentence}")
                counter += 1
    return selectedModel, outputFilePath, textFileText[0], textFileText[1], textFileText[2], translation


with gr.Blocks() as demo:
    # Inputs
    with gr.Row():
        inputVideo = gr.Video(label="Video")
    with gr.Row():
        inputSelectedModel = gr.inputs.Dropdown(
            modelInputs, label="Select your translation model")
    with gr.Column():
        outputAudioFile = gr.File(label="Extracted audio file")
        outputTextFile = gr.File(label="Transcripted text file")
        outputDetectedLanguageText = gr.Textbox(label="Detected language", max_lines=1, lines=1)
        outSelectedModel = gr.outputs.Textbox(label="Selected model")
    with gr.Row():
        outputTranscriptedText = gr.Textbox(label="Transcripted text", lines=10, max_lines=20)
        outputTranslatedText = gr.Textbox(label="Translated text", lines=10, max_lines=20)
    with gr.Row():
        submitButton = gr.Button()
        submitButton.click(fn=mainFunction, inputs=[
            inputSelectedModel, inputVideo], outputs=[outSelectedModel,
                                                      outputAudioFile,
                                                      outputTextFile,
                                                      outputTranscriptedText,
                                                      outputDetectedLanguageText,
                                                      outputTranslatedText])

demo.launch(debug=True)
