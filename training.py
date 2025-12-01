# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./flan-t5-small-finetuned-orderlist-added-json")
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-finetuned-orderlist-added-json")

# Tokenize inputs
tokenized_inputs = tokenizer(
    "Convert this to list: hi, get half a kilo of onions, a dozen eggs, and 250 ml yogurt.",
    return_tensors="pt",
    max_length=128,
    truncation=True,
    padding=True,
)

outputs = model.generate(
    **tokenized_inputs,
    max_new_tokens=128,      # give it room
    do_sample=False,         # deterministic
    num_beams=4,             # safer formatting
    early_stopping=True,
)

decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Generate outputs
# outputs = model.generate(**tokenized_inputs)

# Decode outputs
# decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_outputs)
