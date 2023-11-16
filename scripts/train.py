import os
import json
import time
import wandb
import torch
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMultipleChoice
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

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
cosmos = load_dataset("cosmos_qa")
print(cosmos)
print(cosmos['train'][0])

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForMultipleChoice.from_pretrained(config.model_name)


# Tokenize dataset
def tokenize_function(examples):
    complete_contexts = [[context + ' ' + question] * 4 for context,
                         question in zip(examples['context'], examples['question'])]
    answers = []
    for a0, a1, a2, a3 in zip(examples["answer0"], examples["answer1"], examples["answer2"], examples["answer3"]):
        answers.append([a0, a1, a2, a3])
    complete_contexts = sum(complete_contexts, [])
    answers = sum(answers, [])
    labels = examples["label"]
    tokenized_examples = tokenizer(complete_contexts, answers, truncation=True,
                                   padding='max_length', max_length=config.max_seq_length)
    inputs = {k: [v[i:i+4] for i in range(0, len(v), 4)]
              for k, v in tokenized_examples.items()}
    inputs["labels"] = labels
    return inputs


tokenized_cosmos = cosmos.map(tokenize_function, batched=True)


small_train_dataset = tokenized_cosmos["train"].shuffle(
    seed=42).select(range(1000))
small_eval_dataset = tokenized_cosmos["validation"].shuffle(
    seed=42).select(range(1000))


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


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
    save_total_limit=1
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cosmos["train"],
    eval_dataset=tokenized_cosmos["validation"],
    compute_metrics=compute_metrics,
)

# trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=small_train_dataset,
#    eval_dataset=small_eval_dataset,
#    compute_metrics=compute_metrics,
# )

trainer.train()

# Save model
trainer.save_model(os.path.join(config.output_dir, timestamp))
