from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = "./flan-t5-small-finetuned-orderlist-added-json2"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Add </json> as an extra EOS so generation stops exactly at your closing tag
json_end_id = tokenizer.convert_tokens_to_ids("</json>")
base_eos = model.config.eos_token_id
eos_ids = []
if isinstance(base_eos, int):
    eos_ids.append(base_eos)
elif isinstance(base_eos, (list, tuple)):
    eos_ids.extend(base_eos or [])
if json_end_id is not None:
    eos_ids.append(json_end_id)
eos_ids = list(set(eos_ids))

prompt = (
    "Convert this to a JSON shopping list with fields qty (number), unit, item.\n"
    "Return ONLY inside <json>...</json> using one <item>{...}</item> per product.\n"
    "Text: hi, get half a kilo of onions, a dozen eggs, and 250 ml yogurt."
)

tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
gen = model.generate(
    **tok,
    max_new_tokens=256,
    do_sample=False,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=4,
    eos_token_id=eos_ids,
)
decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
print(decoded)

# Optional: parse items back into a list[dict]
import re, json
m = re.search(r"<json>\s*(.*?)\s*</json>", decoded, re.S)
if m:
    block = m.group(1)
    items = re.findall(r"<item>\s*(\{.*?\})\s*</item>", block, re.S)
    parsed = [json.loads(s) for s in items]
    print(parsed)
