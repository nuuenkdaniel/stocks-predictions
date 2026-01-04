'''
functions that relate to testing/validation 
'''
import torch  
import torch.nn as nn 
import torch.nn.functional as F 
import pandas as pd 
import numpy as np 
from data.tokenizer import detokenize 
from utils import create_path 

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

def extract_test_errors(model, test_loader, tokenizer, fileName):
    '''  
    find the test data examples where the model is making mistakes on 
    
    steps: 
        - run model through the test loader 
        - take the news, prediction, ground truth, and prediction softmax of incorrect examples 
        - return in a csv file 
        
    '''
    model.eval() 
    errors=[] 
    with torch.no_grad(): 
        for ids, labels, lengths in test_loader: 
            outputs= model(ids, lengths)    # batch_size x 3 

            # retrieve prob and prediction 
            probs= F.softmax(outputs, dim=1)    # logits turned into prob 
            
            confidence, preds= torch.max(probs, dim=1) 
            mask = preds != labels 

            conversion= {0: "neutral", 1:"negative", 2:"positive"}

            # sotre all mistakes 
            for i in range(len(mask)): 
                if mask[i]: 
                    format_probs = [f"{p:.2f}" for p in probs[i]]
                    errors.append({
                        "news": ids[i].tolist(),
                        "pred_label": conversion[preds[i].item()], 
                        "true_label": conversion[labels[i].item()],
                        "confidence": format_probs
                    })

    print(f"Number of errors:{len(errors)}")
    # convert to dataframe 
    df = pd.DataFrame(errors)

    df["news"] = df["news"].apply(lambda x:tokenizer.decode(x, skip_special_tokens=True))
    path = create_path(folderName="error_analysis", fileName=fileName)
    df.to_csv(path, index=False, float_format="%.2f")
    
    print("----Error Analysis CSV Created----")
    return df 