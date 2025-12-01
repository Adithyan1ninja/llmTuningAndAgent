# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("./flan-t5-small-finetuned-orderlist")
# model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-finetuned-orderlist")

# inputs = tokenizer("Convert this to list: hey, buy a liter milk, and 10 detergent.", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=32)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

tokenizer = AutoTokenizer.from_pretrained("./flan-t5-small-finetuned-orderlist")
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-finetuned-orderlist")

prompt = "Convert this to list: hey, buy a liter milk, and 10 detergent."

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,      # give it room
    do_sample=False,         # deterministic
    num_beams=4,             # safer formatting
    early_stopping=True,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("RAW:", text)

# Optional: strict parse to verify
parsed = json.loads(text)    # will raise if invalid
print("PARSED:", parsed)
