import torch
import fire
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_squared_error
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        df = pd.read_csv(data_path)
        self.data = df[["source", "label"]].values.tolist()

    @property
    def labels(self):
        return [label for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}


def train_regression_model(pretrained_model, train_file, valid_file, test_file):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, num_labels=1
    )
    model_id = pretrained_model.replace("/", "--")

    train_dataset = CustomDataset(tokenizer, train_file)
    valid_dataset = CustomDataset(tokenizer, valid_file)
    test_dataset = CustomDataset(tokenizer, test_file)

    batch_size = 64
    eval_steps = 2
    print(f"Eval steps: {eval_steps}")
    training_args = TrainingArguments(
        output_dir=f"./results/relevance_{model_id}",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=1,
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        bf16=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(f"./results/relevance_{model_id}")

    # Evaluate the model on the test set
    predictions = trainer.predict(test_dataset)
    mse = mean_squared_error(test_dataset.labels, predictions.predictions.squeeze())
    print(f"Test MSE: {mse}")


if __name__ == "__main__":
    fire.Fire(train_regression_model)
