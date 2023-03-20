from backend.SoundExtractor import *
from backend.SoundTranscriptor import *
#import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

videoFile = "mp4/agriculture.mp4"
audioFile = "mp4/agriculture.mp3"
textFile  = "mp4/agriculture_transcription.txt"

soundExtractor = SoundExtractor()
soundTranscriptor = SoundTranscriptor()
#video to sound
soundExtractor.setVideoFile(videoFile)
soundExtractor.setAudioFile(audioFile)
soundExtractor.extractAudio()
# Audio 2 text
print(f"Available models: {soundTranscriptor.fetchAvailableModel()}")
soundTranscriptor.setAudioFile(audioFile)
soundTranscriptor.setTextFile(textFile)
soundTranscriptor.transcribe()