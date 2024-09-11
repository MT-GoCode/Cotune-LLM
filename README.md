# Cotune-LLM

Running instructions:

Change Data class in `constants.py` to the actual locations of your 4 csv's downloaded from the Kaggle challenge.

Setup environment using 

```
pip install -r requirements.txt
```

## Step 1: make datasets by encoding CTR data with text

```
python workflows.py
```

After running, the following files will be created:

+ Dataset dir: encoded_dataset 
+ Vocabulary file: vocab_map.json

## Step 2: load dataset and vocab created and define custom tokenizer

Run cotune_with_custom_vocab.ipynb
