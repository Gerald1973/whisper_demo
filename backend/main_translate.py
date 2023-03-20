from backend.SoundTranscriptor import *
#Disable https verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

audioFile = "mp4/input.mp3"
textFile  = "mp4/input_transcribed.txt"

soundTranscriptor = SoundTranscriptor()
print(f"Available models: {soundTranscriptor.fetchAvailableModel()}")
# Audio transcribe
# Audio 2 english
soundTranscriptor.setAudioFile(audioFile)
soundTranscriptor.setTextFile(textFile)
print('''
Transcription
=============''')
soundTranscriptor.transcribe()
print('''
Translation
===========''')
soundTranscriptor.translate()