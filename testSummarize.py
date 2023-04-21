from backend.Summarizer import Summarizer

file = open("dataTest/A_PET_READER.txt", "r", encoding="utf-8")
tmpArticle = file.read()
summarizer = Summarizer(tmpArticle)
result = summarizer.summarize()
print(result)
