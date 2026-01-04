'''
functions that relate to testing/validation 
'''
import torch  
import torch.nn as nn 
import time 

def test_loop(test_loader, model, criterion):
    '''  
    function to run tests 
    @param criterion is the loss function used at training time 
    ''' 
    model.eval()    
    avg_test_loss=0 
    with torch.no_grad():  # disable gradient graphs  
        for ids, labels, lengths in test_loader: 
            prediction= model(ids, lengths) # make prediction 
            loss= criterion(prediction, labels) 
            avg_test_loss +=loss.item() 
    return round(avg_test_loss/len(test_loader) ,3) 


