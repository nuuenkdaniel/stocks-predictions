'''
Contains utility/helper functions for different files 
'''
import matplotlib.pyplot as plt  
import torch 
import torch.nn as nn 

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