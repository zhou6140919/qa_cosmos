import sys
import os
import json
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from argparse import ArgumentParser, Namespace


cosmos = load_dataset("cosmos_qa")

parser = ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=True,
    default="config/config.json",
    help="Path to config file"
)
parser.add_argument(
    "--timestamp",
    "-t",
    type=str,
    required=True,
    help="Timestamp of the checkpoint to load"
)
args = parser.parse_args()
config = Namespace(
    **vars(args), **vars(Namespace(**json.load(open(args.config)))))

tokenizer = AutoTokenizer.from_pretrained(config.model_name)


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
                                   padding='max_length', max_length=config.max_seq_length, return_tensors='pt')
    inputs = {k: [v[i:i+4] for i in range(0, len(v), 4)]
              for k, v in tokenized_examples.items()}
    inputs["labels"] = labels
    return inputs


def collator(batch):
    input_ids = torch.stack([torch.tensor(b['input_ids']) for b in batch])
    attention_mask = torch.stack(
        [torch.tensor(b['attention_mask']) for b in batch])
    labels = torch.tensor([b['labels'] for b in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


tokenized_cosmos = cosmos['test'].map(tokenize_function, batched=True)


dataloader = DataLoader(tokenized_cosmos, batch_size=8,
                        shuffle=False, collate_fn=collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForMultipleChoice.from_pretrained(
    os.path.join("checkpoints", args.timestamp))
model.to(device)


with torch.no_grad():
    model.eval()
    preditions = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preditions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# save predictions
with open(os.path.join('checkpoints', args.timestamp, 'predictions.txt'), 'w') as w:
    for p in preditions:
        w.write(str(p) + '\n')
