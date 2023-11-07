import os
import json
import time
import wandb
import torch
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate

timestamp = time.strftime("%Y%m%d_%H%M%S")

# Load config
parser = ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=True,
    default="config/config.json",
    help="Path to config file"
)
args = parser.parse_args()
config = Namespace(
    **vars(args), **vars(Namespace(**json.load(open(args.config)))))
print(config)
wandb.init(project="qa_cosmos", name=timestamp, config=config)

# Load dataset
dataset = load_dataset('csv', data_files={
                       'train': 'datasets/raw_data/train.csv', 'dev': 'datasets/raw_data/valid.csv'})
print(dataset)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    config.model_name, num_labels=4)

# tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
# model.resize_token_embeddings(len(tokenizer))


# Tokenize dataset
def tokenize_function(examples):
    contexts = []
    answers = []
    for c, q, a0, a1, a2, a3 in zip(examples['context'], examples['question'], examples['answer0'], examples['answer1'], examples['answer2'], examples['answer3']):
        context = c + ' ' + q
        answer = '0: ' + str(a0) + ' 1: ' + str(a1) + \
            ' 2: ' + str(a2) + ' 3: ' + str(a3)
        contexts.append(context)
        answers.append(answer)

    inputs = tokenizer(contexts, answers, padding="max_length",
                       max_length=config.max_seq_length, truncation=True)
    inputs["labels"] = examples["label"]
    return inputs


def tokenize_function_test(examples):
    contexts = []
    answers = []
    for c, q, a0, a1, a2, a3 in zip(examples['context'], examples['question'], examples['answer0'], examples['answer1'], examples['answer2'], examples['answer3']):
        context = c + ' ' + q
        answer = '0: ' + str(a0) + ' 1: ' + str(a1) + \
            ' 2: ' + str(a2) + ' 3: ' + str(a3)
        contexts.append(context)
        answers.append(answer)

    inputs = tokenizer(contexts, answers, padding="max_length",
                       max_length=config.max_seq_length, truncation=True)
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

test_dataset = load_dataset("json", data_files='datasets/raw_data/test.jsonl')
test_dataset = test_dataset.map(tokenize_function_test, batched=True)['train']


# small_train_dataset = tokenized_datasets["train"].shuffle(
#     seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["dev"].shuffle(
#     seed=42).select(range(1000))


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Train
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=os.path.join(config.output_dir, timestamp),
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.train_batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    warmup_ratio=config.warmup_ratio,
    run_name=timestamp,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    metric_for_best_model="eval_accuracy",
    learning_rate=config.learning_rate,
    load_best_model_at_end=True,
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2
)

# logging_dir=os.path.join(config.output_dir, timestamp, "logs")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    compute_metrics=compute_metrics,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

trainer.train()

# Predict
predictions = trainer.predict(test_dataset=test_dataset)

# Save predictions
predictions = np.argmax(predictions.predictions, axis=-1)
with open(os.path.join(config.output_dir, timestamp, "predictions.txt"), "w") as w:
    for p in predictions:
        w.write(str(p) + "\n")

# Save model
trainer.save_model(os.path.join(config.output_dir, timestamp))
