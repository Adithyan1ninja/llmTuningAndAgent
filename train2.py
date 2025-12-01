from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

def trainmodel():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # 1) Add special fence tokens (recommended even if your data already has the tags)
    special_tokens = {"additional_special_tokens": ["<json>", "</json>", "<item>", "</item>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 2) Load your dataset (expects records with keys: "input", "output")
    dataset = load_dataset("json", data_files="final.json")

    # 3) Map without re-wrapping (targets already have <json><item>...</item></json>)
    def map_example(batch):
        model_inputs = tokenizer(batch["input"], truncation=True)
        labels = tokenizer(text_target=batch["output"], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # remove original columns to keep Trainer happy
    tokenized = dataset.map(
        map_example,
        batched=True,
        remove_columns=dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    )

    args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-small-finetuned-orderlist-v4",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        logging_dir="./logs",
        predict_with_generate=False,  # training only
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    outdir = "./flan-t5-small-finetuned-orderlist-added-json2"
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)

trainmodel()
