from moviepy.editor import *

class SoundExtractor:
    
    def __init__(this, videoFile, audioFile):
        this.videoFile = videoFile
        this.audioFile = audioFile
    
    def setVideoFile(this, videoFile):
        this.videoFile = videoFile
    
    def setAudioFile(this,soundFile):
        this.audioFile = soundFile

    def extractAudio(this) -> str:
        """_summary_

        Args:
            this (_type_): _description_

        Returns:
            str: The mp3 file path
        """
        inputVideo = VideoFileClip(this.videoFile)
        inputVideo.audio.write_audiofile(this.audioFile)    
        return this.audioFile 