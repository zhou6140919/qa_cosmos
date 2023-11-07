from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

dataset = load_dataset('csv', data_files={
                       'train': 'datasets/raw_data/train.csv', 'dev': 'datasets/raw_data/valid.csv'})
print(dataset)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    lengths = []
    for c, q, a0, a1, a2, a3 in zip(examples['context'], examples['question'], examples['answer0'], examples['answer1'], examples['answer2'], examples['answer3']):
        new_text = c + ' ' + q + ' ' + '0: ' + str(a0) + \
            ' 1: ' + str(a1) + ' 2: ' + str(a2) + ' 3: ' + str(a3)
        tokenized = tokenizer.encode_plus(new_text)
        lengths.append(len(tokenized.input_ids))

    return {'lengths': lengths}


# length statistics
new_dataset = dataset['train'].map(tokenize_function, batched=True)
df = pd.Series(new_dataset['lengths'])
print(df.describe())

new_dataset = dataset['dev'].map(tokenize_function, batched=True)
df = pd.Series(new_dataset['lengths'])
print(df.describe())
