from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import PyPDF2
import torch
import json
#re for regex
import re

class PdfExtractor:

    def __init__(this, pdfFile, textFile):
        this.device = 0 if torch.cuda.is_available() else -1
        this.model_ckpt = "papluca/xlm-roberta-base-language-detection"
        this.pipe = pipeline("text-classification", model=this.model_ckpt, device=this.device)
        this.pdfFile = pdfFile
        this.textFile = textFile

    def setPdfFile(this, pdfFile):
        this.pdfFile = pdfFile

    def setTextFile(this, textFile):
        this.textFile = textFile

    def predict_lang(this, text): 
        result = this.pipe(text)
        label = result[0]['label']
        lang_code = label.split('-')[0]
        return lang_code

    def extractText(this) -> str:
        """_summary_

        Args:
            this (_type_): _description_

        Returns:
            str: The text file path
        """
        pdfFileObj = open(this.pdfFile, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        numPages = len(pdfReader.pages)
        pageNumber = 0
        text = ""
        while pageNumber < numPages:
            pageObject = pdfReader.pages[pageNumber]
            pageNumber += 1
            text += pageObject.extract_text()
        if text != "":
            text = text
        else:
            text = "Empty or malformed PDF file."
        f = open(this.textFile, "w")
        f.write(text)
        f.close()

        tokenizer = AutoTokenizer.from_pretrained(
            "papluca/xlm-roberta-base-language-detection")
        model = AutoModelForSequenceClassification.from_pretrained(
            "papluca/xlm-roberta-base-language-detection")
        inputs = tokenizer(text, return_tensors="pt")
        res = re.sub(r'[^a-zA-Z]', '', text)
        detectedLanguage = this.predict_lang(res[:512])
        print(f"Detected language: {detectedLanguage}")
        print(f"File             : {this.textFile}")
        print(f"Text             : {text}")

        return this.textFile, text, detectedLanguage
