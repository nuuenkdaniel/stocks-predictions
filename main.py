from src.helpers.tokenizer import Tokenizer
from src.helpers.vocab_builder import Vocab_Builder
from src.models.rnn import biGRU

import sys
import pandas as pd
import torch.nn as nn
import torch
import math

def split_dataset(df: pd.DataFrame, data_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle data
    cutoff = int(len(df) * data_split)
    return (df.iloc[:cutoff], df.iloc[cutoff:])

def train(model: biGRU, data: pd.DataFrame, vocab_dict, optimizer: torch.optim.Adam, criterion: nn.CrossEntropyLoss, batch_size: int, device: str) -> tuple[float, float]:
    total_loss = 0
    total_accuracy = 0


    # Batch the input
    for i in range(0, len(data), batch_size):

        # Seperate into batch
        batch = data.iloc[i : i + batch_size]
        batch_ids = []
        batch_labels = []

        for _, row in batch.iterrows():
            # Convert each token into their ids and then convert each sentance into a tensor
            token_ids = [vocab_dict.get(token, vocab_dict["<UNKNOWN>"]) for token in row["tokens"]]
            batch_ids.append(torch.tensor(token_ids, dtype=torch.long))
            batch_labels.append(row["label"])

        # Pad each tensor to make sure they're the same size since sentances have different lengths
        batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=0)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long) # Convert labels to tensors

        # Cache to gpu
        batch_ids = batch_ids.to(device)
        batch_labels = batch_labels.to(device)

        # Training
        optimizer.zero_grad() # Zero out the past gradients
        predictions = model(batch_ids) # Forward pass
        loss = criterion(predictions, batch_labels)

        # Calc acc
        predicted_classes = torch.argmax(predictions, dim=1)
        correct = (predicted_classes == batch_labels).float()
        acc = correct.sum()/len(correct)

        # Update model
        loss.backward() # back pass
        optimizer.step() # Update weights

        total_loss += loss.item()
        total_accuracy += acc.item()

    n_batches = math.ceil(len(data)/batch_size)

    return total_loss/n_batches, total_accuracy/n_batches
        
def evaluate(model: biGRU, data: pd.DataFrame, vocab_dict, criterion: nn.CrossEntropyLoss, batch_size: int, device: torch.device) ->  tuple[float, float]:
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad(): # Disable gradient calculation since we're just evaluation
        # Seperate into batch
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            batch_ids = []
            batch_labels = []

            for _, row in batch.iterrows():
                # Convert each token into their ids and then convert each sentance into a tensor
                token_ids = [vocab_dict.get(token, vocab_dict["<UNKNOWN>"]) for token in row["tokens"]]
                batch_ids.append(torch.tensor(token_ids, dtype=torch.long))
                batch_labels.append(row["label"])

            # Pad each tensor to make sure they're the same size since sentances have different lengths
            batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long) # Convert labels to tensors

            # Cache to gpu
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_ids) # Forward pass
            loss = criterion(predictions, batch_labels)

            # Calc acc
            predicted_classes = torch.argmax(predictions, dim=1)
            correct = (predicted_classes == batch_labels).float()
            acc = correct.sum()/len(correct)

            total_loss += loss.item()
            total_accuracy += acc.item()

    n_batches = math.ceil(len(data)/batch_size)

    return total_loss/n_batches, total_accuracy/n_batches

def usage():
    print("Usage:")
    print("  main.py [FILE]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    file = sys.argv[1]

    DATA_SPLIT=0.8
    EMBED_DIM=64
    HIDDEN_SIZE=64
    N_LAYERS=1
    OUTPUT_DIM=3
    LEARNING_RATE=0.001
    EPOCHS=10
    BATCH_SIZE=64

    # Figure out if my gpu works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Tokenize dataset
    tokenizer = Tokenizer(file)
    df = tokenizer.tokenized_df()
    print(f"Sample size: {len(df)}")

    # Create vocab dict
    vocab_builder = Vocab_Builder()
    vocab_dict = vocab_builder.build(df)
    print(f"Vocab size: {len(vocab_dict)}")

    # Split data
    train_df, eval_df = split_dataset(df, DATA_SPLIT)

    # Model setup
    model = biGRU(len(vocab_dict), EMBED_DIM, HIDDEN_SIZE, N_LAYERS, OUTPUT_DIM)
    model = model.to(device) # Cache in gpu
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"Training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):

        # Train the model 
        model.train() # Sets model to training mode
        df = df.sample(frac=1).reset_index(drop=True) # Shuffle data
        train_loss, train_acc = train(model, train_df, vocab_dict, optimizer, criterion, BATCH_SIZE, device)

        # Validate the model
        model.eval() # Sets model to eval mode
        eval_loss, eval_acc = evaluate(model, eval_df, vocab_dict, criterion, BATCH_SIZE, device)

        print(f"Epoch: {epoch+1}")
        print(f"    Train:      loss={train_loss:.3f} | accuracy={train_acc:.3f}")
        print(f"    Validation: loss={eval_loss:.3f} | accuracy={eval_acc:.3f}")
