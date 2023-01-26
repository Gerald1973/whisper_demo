import os
import openai
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "Resume moi le texte suivant :"

model = "text-davinci-002"
completions = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)

message = completions.choices[0].text
print(message)