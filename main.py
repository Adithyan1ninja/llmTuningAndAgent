# serve_quantized.py
import os, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "google/flan-t5-small"
SAVE_DIR = "./flan_t5_small_int8_artifacts"
PT_PATH  = f"{SAVE_DIR}/flan_t5_small_int8.pt"

os.makedirs(SAVE_DIR, exist_ok=True)

if os.path.exists(PT_PATH):
    model = torch.load(PT_PATH, map_location="cpu").eval()
    tok = AutoTokenizer.from_pretrained(SAVE_DIR)
    print("Loaded cached quantized model.")
else:
    print("Quantizing (first run)…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).eval()
    model = torch.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8).eval()
    torch.save(model, PT_PATH)
    tok.save_pretrained(SAVE_DIR)
    print("Saved quantized model for future runs.")

def infer(prompt: str):
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model.generate(
            **inputs, max_new_tokens=32, min_new_tokens=1,
            do_sample=False, num_beams=1, use_cache=True,
            no_repeat_ngram_size=3, repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
        )
        dt = time.perf_counter() - t0
    print(tok.decode(out[0], skip_special_tokens=True))
    print(f"Elapsed: {dt:.3f}s")

infer("Instruction: Reply briefly.\nUser:CAN WE EAT BANANA?\nAssistant:")
