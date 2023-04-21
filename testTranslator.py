from backend.HelsinkiTranslator import HelsinkyTranslator
from backend import utils


print("Testing HelsinkiTranslator.py")
print("==============================")

file = open("dataTest/A_PET_READER.txt", "r")
tmpArticle = file.read()
translator = HelsinkyTranslator(tmpArticle, "en", "fr")
modelName = translator.getModelName()

# Test fetchByNumberOfTokens
print("Test fetchByNumberOfTokens")
print("==============================")
results = utils.segmentize(modelName, tmpArticle, 512)
print(results)
print("==============================")

print ("Test translate")
print("==============================")
results = translator.translate()
print(results)
print("==============================")
