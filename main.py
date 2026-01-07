''' 
main function to execute code
'''
from train import train_loop 
import settings 
import os 
import time 
import model.tokenizers as tokenizers 
from data.data_load import load_data, process_dataset, create_loaders  
from data.feature_pruning import handle_pruning
from evaluator import k_fold_validation 
import settings

# disable parallelization of tokenizer for DataLoader parallelization 
os.environ["TOKENIZERS_PARALLELISM"]= "false"  

def main(tokenizer_model, 
         pretrain_embed= None, 
         cross_validate=False,
         pruning= False, 
         save_weights= False, 
         ): 
    
    start_time= time.time() 
    
    settings.hyperparams["tokenizer_model"]= tokenizer_model 

    # create the data 
    # (token_ids, labels, lengths) 
    # batch_size x max_seq_len 
    df,_ = load_data() 

    # pruning 
    if (pruning): 
        feature_path = handle_pruning(df, k=settings.hyperparams["k_pruning"])
        settings.hyperparams["selected_feature"]= feature_path 

    dataset= process_dataset(df, model=tokenizer_model) 
    vocab_size = dataset.get_vocab_size() 
    print(f"Tokenizer model: {tokenizer_model} | Vocab Size: {vocab_size}")

    settings.hyperparams["vocab_size"]= vocab_size
    settings.hyperparams["tokenizer_model"]= tokenizer_model
    settings.hyperparams["pretrain_embed"] = pretrain_embed

    if (cross_validate):
        train_loss, train_acc, test_loss, test_acc= k_fold_validation(dataset) 
    
    else: 
        train_loader, test_loader, train_size, test_size = create_loaders(dataset,
                                                                          batch_size=settings.hyperparams["batch_size"])
        
        train_loss, train_acc, test_loss, test_acc= train_loop(
                    train_loader=train_loader, test_loader=test_loader, train_size=train_size, test_size=test_size,
                        save_weights=save_weights   
                    )  

    end_time= time.time() 
    print(f"Total Program Execution Time (min): {round((end_time-start_time)/60, 3)}")

if __name__ =="__main__":
    main(tokenizer_model=tokenizers.models["bert-cased"], 
        pretrain_embed=tokenizers.models["bert-cased"],
        cross_validate=True   
          ) 