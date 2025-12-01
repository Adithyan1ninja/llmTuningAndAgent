from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

def trainmodel():


    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    dataset = load_dataset("json", data_files="restuctured_for_training.json")
    tokenized = dataset.map(
        lambda x: tokenizer(x["input"], text_target=x["output"], truncation=True),
        batched=True
    )

    args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-small-finetuned-orderlist-v3",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_strategy="epoch",   # 👈 ensure saving works
        logging_steps=1,
        save_total_limit=1,
        logging_dir="./logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model("./flan-t5-small-finetuned-orderlist-added-json")  # ✅ must include this explicitly
    tokenizer.save_pretrained("./flan-t5-small-finetuned-orderlist-added-json")  # ✅ also save tokenizer

trainmodel()