from transformers import PreTrainedTokenizer

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, **kwargs):
        self.vocab = vocab
        self.ids_to_tokens = {i: token for i, token in enumerate(self.vocab)}
        self.tokens_to_ids = {token: i for i, token in enumerate(self.vocab)}

        super().__init__(**kwargs)

    def _tokenize(self, text):
        tokens = []
        for part in text.split():
            if part.isdigit():
                # Split digits into separate tokens
                tokens.extend(list(part))
            else:
                tokens.append(part)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.tokens_to_ids[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens[_id] for _id in ids]

    def _convert_token_to_id(self, token):
        return self.tokens_to_ids.get(token, self.tokens_to_ids.get("[UNK]"))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "[UNK]")

    def get_vocab(self):
        return self.tokens_to_ids


import json

with open("vocab_map.json", "r") as f:
    vocab = json.load(f)
tokenizer = CustomTokenizer(vocab=vocab)

rows = [line for line in open("throwaway_outputs/test_quantize.txt", "r")]

tokens = tokenizer.tokenize(rows[0])
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
