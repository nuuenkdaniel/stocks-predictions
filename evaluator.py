'''
functions that relate to testing/validation & ablation studies  
'''
import torch  
import torch.nn.functional as F 
import pandas as pd 
from utils import create_path 
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
import settings 

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


def k_fold_validation(dataset:Dataset, k=10): 
    from train import train_loop
    from data.custom_loader import collate_fn 
    '''  
    implementation of the k-fold cross-validation 

    goal: find the best hyperparameters to use that can produce the best model 

    After K-fold cross validation, we get better sense of the model's true performance at specific hyperparam setting (model won't memorize based on the specific order of training data)
    This gives better approximation of how our model will perform with future data (after deployment), if they come from the same distribution as the data used for training and testing 
    Becauese of Law of Large Numbers, this approximates the average mean the model sees 

    Splitting: 
        - splitting is based on the indices, use library to create those indices
        - create corresponding train and test loaders 
    '''
    seed= settings.seed 

    # splitting by position in the document 
    # kf= KFold(n_splits=k, shuffle=True, random_state=seed)

    # splitting by taking into account of labels (more diverse)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    best_train_loss=0 
    best_train_acc=0 
    best_test_loss =0 
    best_test_acc=0 

    for fold, (train_ids, val_ids) in enumerate(skf.split(dataset, dataset.labels)):
        print(f"------------FOLD {fold + 1}-----------")

        # create each subset for training and eval 
        train_sub = Subset(dataset, indices=train_ids)
        val_sub = Subset(dataset, indices=val_ids)

        train_size= len(train_sub)
        test_size = len(val_sub)

        # create data loader 
        train_loader = DataLoader(train_sub, batch_size=settings.hyperparams["batch_size"], shuffle=True,num_workers=2, collate_fn=collate_fn)
        test_loader= DataLoader(val_sub, batch_size=settings.hyperparams["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_fn)

        train_loss, train_acc, test_loss, test_acc= train_loop(
                    train_loader=train_loader, test_loader=test_loader, train_size=train_size, test_size=test_size)
        
        if (test_acc>best_test_acc):
            best_train_loss= train_loss 
            best_train_acc= train_acc
            best_test_loss= test_loss 
            best_test_acc= test_acc

    print(f"Final Statistics: \n\tTrain Loss:{best_train_loss} Train Acc: {best_train_acc} | Test Loss: {best_test_loss} Test Acc: {best_test_acc}")
    return best_train_loss, best_train_acc, best_test_loss, best_test_acc 