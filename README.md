# Homework 3 Multi Choice Question Answering

## Dataset
CosMos QA

## Preprocessing

```bash
conda create -n cosmos_qa python=3.11
pip install -r ./requirements.txt
```

### Length Statistics

```bash
bash ./statistics.sh
```

```bash
Train set
count    25262.000000
mean       127.104584
std         29.215060
min         52.000000
25%        105.000000
50%        124.000000
75%        146.000000
max        344.000000

Dev set
count    2985.000000
mean      138.545394
std        30.169795
min        69.000000
25%       116.000000
50%       135.000000
75%       157.000000
max       264.000000
```

## Training

Edit hyperparameters in `./config/config.json`

```bash
bash ./train.sh
```

## Testing

```bash
bash ./test.sh
```

The predictions are in `predictions.txt` file in the model path of checkpoints folder
