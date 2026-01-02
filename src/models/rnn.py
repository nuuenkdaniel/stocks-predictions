from src.helpers.tokenizer import Tokenizer
from src.helpers.vocab_builder import Vocab_Builder

import sys
import pandas as pd
import torch
import torch.nn as nn

class biGRU(nn.Module):
    """Initializing function
    Params:
        vocab_size (int): number of unique words to be referenced 
        embedded_dim (int): arbitrary number for complexity of a word
        hidden_size (int): internal memory, how many features it tracks
        output_dim (int): number of possible outputs, the end result
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int, output_dim: int):
        super().__init__()
        # Converts vocab ID's to a lookup table consiting of a vector for each ID
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Defining a bidirectional gru neural network
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)

        # Maps the input dimension to the output dimension
        # The input size is multiplied by 2 since the nn is bidirectional
        self.fc = nn.Linear(hidden_size*2, output_dim)

    """Define the forward pass function
    Params:
        text ([[int]]): The batch of tokens from the dataset
    """
    def forward(self, text):
        # Convert text batch into lookup table of vectors
        embedded = self.embedding(text)

        # Get the output (Output at each step) and the hidden (the final output)
        output, hidden = self.gru(embedded)

        # Get the forward hidden state
        hidden_for = hidden[-2, :, :]

        # Get the backward hidden state
        hidden_back = hidden[-1, :, :]

        # Concatenate the two summaries into one vector
        cat_hidden = torch.cat((hidden_for, hidden_back), dim=1)

        # Make final prediction [Negative, Neutral, Positive]
        return self.fc(cat_hidden)

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
