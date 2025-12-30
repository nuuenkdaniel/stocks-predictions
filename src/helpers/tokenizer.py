import sys
import pandas as pd
import re


# Expects CSV
class Tokenizer:
    def __init__(self, file):
        self.file = file

    def clean_data(self)-> pd.DataFrame:
        df = pd.read_csv(self.file, encoding='ISO-8859-1', header=None, names=['label', 'headline'])
        df = df.dropna().reset_index(drop=True)
        df["label"] = df["label"].map({'negative': 0, 'neutral': 1, 'positive': 2}) # type: ignore
        return df

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def tokenized_df(self) -> pd.DataFrame:
        df = self.clean_data()
        df["tokens"] = df["headline"].apply(self._tokenize)
        return df

def usage():
    print("Usage:")
    print("  ./tokenizer [FILE]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        exit(1)

    file = sys.argv[1]
    tokenizer = Tokenizer(file)
    print(tokenizer.tokenized_df())
