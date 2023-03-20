# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

input_text = '''translate English to French :

'''
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids,max_length=4096)
print(tokenizer.decode(outputs[0]))

summarizer = pipeline(task="summarization")
summarizer(input_text)