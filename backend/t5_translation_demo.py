import transformers
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import sys

T5_MODELS = ("t5-small", "t5-base", "t5-large")
model_name = T5_MODELS[0]

def build_model(model_name: str):
    return TFT5ForConditionalGeneration.from_pretrained(model_name)

def build_tokenizer(model_name: str):
    # set logging level to error to prevent unnecessary logging
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    # load from hugging face
    return T5Tokenizer.from_pretrained(model_name)

model = build_model(model_name=model_name)
tokenizer = build_tokenizer(model_name=model_name)

def translate(article: str, srcLanguage: str, toLanguage: str, tokenizer: any, model) -> str:
    str = f"translate from  {srcLanguage} to {toLanguage} : {article}"
    print(str)
    inputs = tokenizer(str,
                       return_tensors="tf", max_new_tokens=1000).input_ids
    result = model.generate(inputs)
    return tokenizer.decode(result[0])


DEFAULT_PHRASE = "The Wheel of Time is an incredible Book."


print(translate(DEFAULT_PHRASE, "English", "Bulgarian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Croatian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Czech", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Danish", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Dutch", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "English", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Estonian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Finnish", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "French", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Greek", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "German", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Hungarian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Irish", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Italian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Latvian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Lithuanian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Maltese", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Polish", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Portuguese", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Romanian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Slovak", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Slovenian", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Spanish", tokenizer, model))
print(translate(DEFAULT_PHRASE, "English", "Swedish", tokenizer, model))
