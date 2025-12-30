'''
Training Loop/main function of the training process 
Bridge design pattern :- 

Training takes a few steps: 
    1. load the data into DataLoader (make them into trainable formats)
    2. initialize the model with hyperparameters for training 
    3. establish training loop to train the model with the data provided   
        - provide loss functions to update model parameters 
        - log/store training related information that you want 
    4 (Optional). Save model weights based for future inferences (load pretrained model is done this way)
'''
import os 
# disable parallelization of tokenizer for DataLoader parallelization 
os.environ["TOKENIZERS_PARALLELISM"]= "false"  

from data.data_load import data_process 
from model.rnn import LSTM
import torch  
import torch.nn as nn 
import time 

# here holds the hyperparameters for our model (global variable for our design)
# these parameters are mainly for model training (constructor params, training loop)
hyperparams= {
    "n_layers": 2, 
    "hidden_dim": 128,  # usually multiple of 64 
    "embed_dim": 100, 
    "output_dim": 3,    # we have 3 classes of labels to predict (neutral, pos, neg)
    "epochs": 5,        # number of times the model trains over the entire set 
    "batch_size": 32,   # batch size of each data training batch 
    "learning_rate": 0.001,  # learning rate of gradient descent 

}

''' 
hidden dim, embed_dim and n_layer determine our model complexity (same as # of trainable parameters the model has)
higher complexity gives model more "power" to capture information and perform better 
but with smaller amount of data, it's a tradeoff between overfitting and underfitting. 
Overfitting is when we have big model/a lot of training for small amount of data, the model can memorize/create large weights in our parameters to fit each data point better (high variance)
underfitting is when the model is too simple to fit all the data (think of a straight line going across data split at different locations of this linear line), it doesn't perform well. 
Another way of looking at underfitting is our likelihood approximation doesn't match the correct Gaussian distribution that generates this data (high bias)
'''




def train_loop(): 
    # create the data 
    # (token_ids, labels, lengths) 
    # batch_size x max_seq_len 
    start_time= time.time() 
    train_loader, _, vocab_size= data_process(batch_size=hyperparams["batch_size"])
    print("-----loader created-----")

    # intialize model 
    train_start= time.time() 
    model = LSTM(n_layers=hyperparams["n_layers"],
                embed_dim=hyperparams["embed_dim"],
                hidden_dim=hyperparams["hidden_dim"],
                output_dim=hyperparams["output_dim"],
                vocab_size=vocab_size)
    
    print(f"Model trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # create optimizer 
    optimizer= torch.optim.Adam(params=model.parameters(), lr=hyperparams["learning_rate"])
    criterion= nn.CrossEntropyLoss() 
    
    # training 
    for i in range(hyperparams["epochs"]):
        model.train()   # start training mode 
        epoch_loss=0 

        for ids, labels, lengths in train_loader: 
            optimizer.zero_grad()  # clear gradients to recalculate new gradients 
            prediction= model(ids, lengths) # make prediction 
            loss= criterion(prediction, labels)     # evaluate loss 

            epoch_loss+= loss.item()   

            loss.backward()  # compute gradient 
            optimizer.step()    # upate parameters 

        
        print(f"Epoch {i+1} | Loss: {round(epoch_loss/(len(train_loader)), 3)}")

    train_end= time.time()  

    end_time= time.time() 
    
    print(f"Total Training Time (min): {round((train_end-train_start)/60, 3)}")
    print(f"Total Program Execution Time (min): {round((end_time-start_time)/60, 3)}")


if __name__ == "__main__":
    train_loop() 