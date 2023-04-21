from backend.PdfExtractor import PdfExtractor


pdfExtractor = PdfExtractor(
    "dataTest/three-musketeers.pdf", "dataTest/three-musketeers.txt")

print("Testing PdfExtractor.py")
print("==============================")
results = pdfExtractor.extract()
print("Text file path:")
print("==============================")
print(results[0])
print("Extracted text:")
print("==============================")
print(results[1])
print("Language:")
print("==============================")
print(results[2])
