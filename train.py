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

from data.data_load import data_process, load_data 
from model.rnn import LSTM
import torch  
import torch.nn as nn 
import time 
from utils import * 
from evaluator import test_loop 
from data.feature_pruning import chi_square_pruning, save_selected_features, prune_dataset
import model.tokenizers as tokenizers 

# here holds the hyperparameters for our model (global variable for our design)
# these parameters are mainly for model training (constructor params, training loop)
hyperparams= {
    "n_layers": 1, 
    "hidden_dim": 32,  # usually multiple of 2 
    "embed_dim": 32, 
    "output_dim": 3,    # we have 3 classes of labels to predict (neutral, pos, neg)
    "epochs": 8,        # number of times the model trains over the entire set 
    "batch_size": 64,   # batch size of each data training batch 
    "learning_rate": 0.001,  # learning rate of gradient descent 
    "dropOut": 0.5,     # reduce overfitting by randomly setting neurons to 0 weight 
    "weight_decay": 1e-5, 
    "l1_lambda": 0.000001, 
    "model": "LSTM"
}

''' 
hidden dim, embed_dim and n_layer determine our model complexity (same as # of trainable parameters the model has)
higher complexity gives model more "power" to capture information and perform better 
but with smaller amount of data, it's a tradeoff between overfitting and underfitting. 
Overfitting is when we have big model/a lot of training for small amount of data, the model can memorize/create large weights in our parameters to fit each data point better (high variance)
underfitting is when the model is too simple to fit all the data (think of a straight line going across data split at different locations of this linear line), it doesn't perform well. 
Another way of looking at underfitting is our likelihood approximation doesn't match the correct Gaussian distribution that generates this data (high bias)
'''

def train_loop(test=False, 
            validate_epoch=False,
            plot_train_loss=False, 
            plot_test_loss= False, 
            save_weights=False,
            tokenizer_model= "bert-base-cased",
            pretrain_embed= None, 
            pruning=False 
            ): 
    '''  
    @param test: if we do testing after the training is completed (one time) 
    @param validate_epoch: validate after each epoch to see the change in performance (measure the bias variance)
    @param save_weights: save the weights of the trained model 
    @param plot_train_loss: show the change in our loss function 
    @param plot_test_loss: plot the change in test loss
    '''
   
    # create the data 
    # (token_ids, labels, lengths) 
    # batch_size x max_seq_len 
    start_time= time.time() 
    df = load_data() 

    # pruning 
    if (pruning): 
        selected_features=chi_square_pruning(df)
        df= prune_dataset(df, selected_features)
        hyperparams["selected_feature"]= ""
    

    train_loader, test_loader, vocab_size, train_size, test_size= data_process(df, batch_size=hyperparams["batch_size"], model=tokenizer_model)
    print(f"Train loader {len(train_loader)} batches | Test loader {len(test_loader)} batches | Batch Size: {hyperparams['batch_size']}")

    hyperparams["vocab_size"]= vocab_size 
    hyperparams["tokenizer_model"]= tokenizer_model 

    embed_weights=None 
    if (pretrain_embed=="bert-base-cased"): 
        print(f"Using saved embedding weights: {pretrain_embed}")
        embed_weights= extract_bert_weight()


    # intialize model 
    if (hyperparams["model"]=="LSTM"): 
        model = LSTM(n_layers=hyperparams["n_layers"],
                    embed_dim=hyperparams["embed_dim"],
                    hidden_dim=hyperparams["hidden_dim"],
                    output_dim=hyperparams["output_dim"],
                    vocab_size=vocab_size,
                    dropOut=hyperparams["dropOut"],
                    embedding_weights=embed_weights)
    

    print(f"Hyperparameters: \n{hyperparams}")
    print(f"Model trainable parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # create optimizer 
    optimizer= torch.optim.Adam(params=model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
    criterion= nn.CrossEntropyLoss() 
    
    # training 
    for i in range(hyperparams["epochs"]):
        epoch_loss=0 
        correct =0 
        model.train()   # start training mode  
        for ids, labels, lengths in train_loader:
            optimizer.zero_grad() 
            prediction= model(ids, lengths) 

            # accuracy calculation 
            _, predicted= torch.max(prediction, dim=1)
            correct += (predicted==labels).sum().item() 
            
            loss= criterion(prediction, labels)   
            
            # L1 norm 
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # loss += l1_norm * hyperparams["l1_lambda"]

            epoch_loss+= loss.item()   

            loss.backward() 
            optimizer.step()   
        train_acc= round(correct/train_size, 3)

        if (validate_epoch):
            test_loss = test_loop(test_loader, model, criterion)
            test_acc= compute_accuracy(model, test_loader)
            print(f"Epoch {i+1}:\n\tAvg Train Loss: {round(epoch_loss/(len(train_loader)), 3)} | Train Accuracy: {train_acc} \n\tAvg Test Loss: {test_loss} | Test Accuracy: {test_acc}")
        else: 
            print(f"Epoch {i+1}:\n\tAvg Train Loss: {round(epoch_loss/(len(train_loader)), 3)} | Train Accuracy: {train_acc}")
            
    
    if (test and not validate_epoch):
        test_loss= test_loop(test_loader, model, criterion)
        test_acc= compute_accuracy(model, test_loader)
        print(f"Test Loss: {test_loss} | Test Accuracy: {test_acc}")

    if (save_weights):
        timestamp = get_timestamp() 
        save_model(model, hyperparams, pretrain_embed, test_acc,
                name=f"{hyperparams["epochs"]} Epochs {hyperparams['model']} Test Acc={test_acc} Train Acc= {train_acc} | {timestamp}.pt") 

    end_time= time.time() 
    print(f"Total Program Execution Time (min): {round((end_time-start_time)/60, 3)}")


if __name__ == "__main__":
    train_loop(validate_epoch=True,
               pruning=True, 
                tokenizer_model=tokenizers.models["bert-cased"], 
                pretrain_embed=tokenizers.models["bert-cased"], 
                save_weights=True 
                ) 