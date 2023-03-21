import whisper
import torch
from . import utils

class SoundTranscriptor():
   
    def __init__(this, audioFile: str, textFile: str):
        """_summary_

        Args:
            this (_type_): _description_
            audioFile (str): audioFile path
            textFile (str): translated file path
        """
        this.audioFile = audioFile
        this.outputFileName = textFile

    def setAudioFile(this, audioFile):
        this.audioFile = audioFile

    def setTextFile(this, textFile):
        this.outputFileName = textFile

    def transcribeByDefault(this):
        model = whisper.load_model("small")
        whisper.load_audio(this.audioFile)
        result = model.transcribe(this.audioFile)
        print(f'Transcription : {result["text"]}')

    def transcribe(this)-> tuple[str, str, str]:
        """_summary_

        Args:
            this (_type_): _description_

        Returns:
            tuple[str, str, str]: 0 => outputFileName, 1 => transcription, 2 => detectedLanguage
        """
        if torch.cuda.is_available():
            print("CUDA is available, we will use the GPU.")
            DEVICE = "cuda"
        else:
            DEVICE = "CPU"
        #model = whisper.load_model("medium", device=DEVICE)
        model = whisper.load_model("large", device=DEVICE)
        audiofull = whisper.load_audio(this.audioFile)
        audio = whisper.pad_or_trim(audiofull)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        for prob in probs:
            print(f'{prob} : {round(probs[prob],5)}')
        detectedLanguage = max(probs, key=probs.get)
        print(f"==> Detected language: {detectedLanguage}")
        # options = whisper.DecodingOptions(
        #     language=detectedLanguage,
        #     without_timestamps=True, fp16=False)
        # result = whisper.decode(model, mel, options)
        options = {
            "task": "transcribe",
            "language": detectedLanguage
        }
        result = whisper.transcribe(
            model=model, audio=audiofull, verbose=False, **options)
        transcription = result["text"]
        print(transcription)
        utils.writeFile(this.outputFileName, transcription)
        return this.outputFileName, transcription, detectedLanguage

    def translate(this):
        if torch.cuda.is_available():
            print("CUDA is available, we will use the GPU.")
            DEVICE = "cuda"
        else:
            DEVICE = "CPU"
        model = whisper.load_model("large-v2", device=DEVICE)
        audiofull = whisper.load_audio(this.audioFile)
        audio = whisper.pad_or_trim(audiofull)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        for prob in probs:
            print(f'{prob} : {round(probs[prob],5)}')
        detectedLanguage = max(probs, key=probs.get)
        print(f"==> Detected language: {detectedLanguage}")
        options = {
            "task": "translate",
            "language": detectedLanguage
        }
        result = whisper.transcribe(
            model=model, audio=audiofull, verbose=False, **options)
        print(result["text"])

    def fetchAvailableModel(this)->list[str]:
        return whisper.available_models()
