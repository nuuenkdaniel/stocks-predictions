
# for sentiment analysis, RNN is many to one (many words to one label of sentiment)
# supervised learning because we have corresponding label to each text 

import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module): 
    # constructor, take the hyperparameters needed for the layers 
    def __init__ (self, 
                n_layers: int,  # number of layers at each time step 
                embed_dim: int,     # embedding dimension to map word to number 
                vocab_size: int,    # number of words in each inference to become number 
                hidden_dim: int,    # hidden dimension (how much information the model can capture)
                output_dim: int,     # the dimension of output (based on what we are using the model for)
                dropOut=0, 
                embedding_weights=None): 
        super().__init__() 


        self.n_layers= n_layers 

        # embedding layer 
        # convers the words to a numerical value for model to process 
        # trained with the model 
        # vocab_size is the size of our dictionary, so this is alike a lookup table, where at each input data, the words become embedded 
        # this is a look up table where the input is an one-hot encoding (index), and nn.Embedding retrieves its embedding dimension 
        self.embedding = nn.Embedding(vocab_size, embed_dim)


        # manually apply dropout 
        if (n_layers==1):
            self.dropOut = nn.Dropout(p=dropOut)
            dropOut=0 

        
        # model inference 
        # start with the embed dimension and start running the model on (matrix multiplication)
        # each timestep has num_layers number of hidden layers, each hidden layer is of size hidden_dim 
        # batch_size x seq_len x hidden_dim 
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True, 
                            bias=True,
                            dropout=dropOut, 
                            bidirectional=True 
                            )
        
        

        # using embedding weights for tokens 
        if (embedding_weights is not None):
            self.embedding= nn.Embedding.from_pretrained(embedding_weights, freeze=False)
            self.lstm = nn.LSTM(embedding_weights.shape[1], 
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True, 
                            bias=True,
                            dropout=dropOut, 
                            bidirectional=True 
                            )
        
        # fully connected network to finalize our predictions 
        # map the hidden layer output to diffrent outputs (neutral, postive, negative)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    # inference (running the model here)
    def forward(self, ids, lengths):
        '''  
        forward does the inference of the model 
        we will input the tokenized inputs from the tokenizer from different batches 
            - embedded into hidden dimension to capture semantic information 
            - padded to make every sentence in the batch into the same size (matrix multiplication requires a rectangle shape)
        
        @param ids: padded_token_ids of the news (batch_size x max_seq_length)
        @param lengths: original length of the news before padding 

        Padding: 
            - each sentence becomes the same size (largest sentence of the batch) 
            - [We, want, cake, PAD=0, PAD=0] 
            - during processing by the LSTM, the padding will be removed, otherwise the LSTM will process the 0s, causing incorrect outputs 
        '''
        # first embed the input token ids 
        embedded =self.embedding(ids)   # (batch_size, max_seq_len, embed_dim)
        

        # pack the inputs so that LSTM can process and know to stop before the PAD 
        # saves computational resources because the PAD of each sentence is not processed 
        padded_embed = pack_padded_sequence(embedded, 
                                            lengths.cpu(),  # bring to cpu to be processable 
                                            batch_first=True,
                                            enforce_sorted=False)


    
        # pass through the lstm network 
        # lstm_out is the hidden states of the last layer of every timestep (good for Many to Many)
        # hidden is the final hidden states of every layer from the last timestep (after model processed the entire input)
        # cell is the cell-state (internal cells) of each layer 
        padded_lstm_out, (hidden, cell) = self.lstm(padded_embed)

        # get the last hidden state to do inference 
        # combine birectional states 
        combined = torch.cat( (hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        if (self.n_layers)==1: 
            combined= self.dropOut(combined) 

        # do the final prediction (map the last layer)
        return self.fc(combined)