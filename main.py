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


    if (cross_validate):
        pass 
    else: 
        train_loader, test_loader, train_size, test_size = create_loaders(dataset,
                                                                          batch_size=settings.hyperparams["batch_size"])
        
        _, train_acc, _, test_acc= train_loop(
                    train_loader=train_loader, test_loader=test_loader, train_size=train_size, test_size=test_size, vocab_size=vocab_size,
                        tokenizer_model=tokenizer_model,
                        pretrain_embed=pretrain_embed, 
                        save_weights=save_weights   
                    )  


    end_time= time.time() 
    print(f"Total Program Execution Time (min): {round((end_time-start_time)/60, 3)}")

if __name__ =="__main__":
    main(tokenizer_model=tokenizers.models["bert-cased"], 
        pretrain_embed=tokenizers.models["bert-cased"],
          ) 