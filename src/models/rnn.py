from src.helpers.tokenizer import Tokenizer
from src.helpers.vocab_builder import Vocab_Builder

import sys
import pandas as pd

def usage():
    print("Usage:")
    print("  ./rnn.py [FILE]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        exit(1)

    file = sys.argv[1]
    tokenizer = Tokenizer(file)
    print(list(Vocab_Builder.build(tokenizer.tokenized_df()))[:5])
