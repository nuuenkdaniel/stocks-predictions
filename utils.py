'''
Contains utility/helper functions for different files 
'''
import matplotlib.pyplot as plt  
import torch 
import torch.nn as nn 
from transformers import BertModel
from datetime import datetime 
import os 

def plot_train_loss(losses): 
    '''  
    plot the training loss at different epochs 
    '''
    pass  



def compute_accuracy(model,loader):
    '''  
    calculate the accuracy of the model over a dataset 

    steps: 
        - loop through the loader 
        - compute model output and get the prediction's index (which index has the highest prob)
        - compare with the label 
    '''
    model.eval() 
    correct =0 
    total =0 
    with torch.no_grad(): 
        for ids, labels, lengths in loader: 
            outputs= model(ids, lengths)

            # retrieve the highest raw value index (same as highest prob index )
            _, predicted= torch.max(outputs, dim=1)
            total+= labels.size(0)
            correct += (predicted==labels).sum().item() 
        return round(correct/total, 3)
    

def extract_bert_weight(model="bert-base-cased"):
  
    bert_model = BertModel.from_pretrained(model)

    # [30522, 768]
    # Torch Tensor 
    bert_weights = bert_model.embeddings.word_embeddings.weight.data.clone()

    del bert_model

    return bert_weights 


def save_model(model, config,pretrain_embed, test_acc, name): 
    '''  
    save model weights with all the hyperparams 
    '''
    checkpoint= {
        "state_dict": model.state_dict(), 
        "config": config, 
        "pretrain_embed": pretrain_embed, 
        "test_acc": test_acc
    }
    path = create_path(folderName="checkpoints", fileName=name)
    torch.save(checkpoint, path)
    print("-----Model Saved-----")


def create_path(folderName, fileName):
    # create the folder 
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    # create path 
    path = os.path.join(folderName, fileName) 

    return path 



def load_checkpoint(path):
    return torch.load(path)

def get_timestamp(): 
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    return timestamp

def calcualte_epoch(E_old, N_old, N_new): 
    '''  
    find number of epochs for training while keeping the number of total iterations the same 

    E_new = E_old * (Number of samples trained before)/ (Number of samples to be trained)
    '''
    pass 


if __name__ == "__main__":
    pass 