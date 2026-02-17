# train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Example: load a toxicity dataset (placeholder)
    dataset = load_dataset("imdb")  # Replace with real moderation dataset

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./moderation_model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    model.save_pretrained("./moderation_model")
    tokenizer.save_pretrained("./moderation_model")

if __name__ == "__main__":
    main()
