from soundextractor import *
from soundtranscriptor import *
#import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

videoFile = "mp4/input.mp4"
audioFile = "mp4/input.mp3"
textFile  = "mp4/input_transcribed.txt"

soundExtractor = SoundExtractor()
soundTranscriptor = SoundTranscriptor()
#video to sound
soundExtractor.setVideoFile(videoFile)
soundExtractor.setAudioFile(audioFile)
soundExtractor.extractAudio()
# Adio 2 text
print(f"Available models: {soundTranscriptor.fetchAvailableModel()}")
soundTranscriptor.setAudioFile(audioFile)
soundTranscriptor.setTextFile(textFile)
soundTranscriptor.transcribe()