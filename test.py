from backend.HelsinkiTranslator import HelsinkyTranslator


print("Testing HelsinkiTranslator.py")
print("==============================")

file = open("test4translate.txt", "r")
tmpArticle = file.read()
translator = HelsinkyTranslator(tmpArticle, "en", "fr")
modelName = translator.getModelName()

# Test fetchByNumberOfTokens
print("Test fetchByNumberOfTokens")
print("==============================")
results = translator.segmentize(modelName, tmpArticle, 256)
print(results)
print("==============================")

print ("Test translate")
print("==============================")
results = translator.translate()
print(results)
print("==============================")
