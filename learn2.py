from transformers import pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
print(pipe("Translate to French: Hello, how are you?")[0]["generated_text"])
