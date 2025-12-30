import pandas as pd

class Vocab_Builder:
    @staticmethod
    def build(data: pd.DataFrame):
        vocab = {"<UNKNOWN>": 0}
        for tokens in data["tokens"]:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab
