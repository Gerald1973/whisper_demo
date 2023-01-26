from moviepy.editor import *

class SoundExtractor:
    def __init__(this):
        this.videoFile = ""
        this.audioFile = ""

    def setVideoFile(this, videoFile):
        this.videoFile = videoFile
    
    def setAudioFile(this,soundFile):
        this.audioFile = soundFile

    def extractAudio(this):
        inputVideo = VideoFileClip(this.videoFile)
        outputAudio = inputVideo.audio.write_audiofile(this.audioFile)


    

    


        