''' 
main function to execute code
'''
from train import train_loop 
import settings 
import os 
import time 
import model.tokenizers as tokenizers 

# disable parallelization of tokenizer for DataLoader parallelization 
os.environ["TOKENIZERS_PARALLELISM"]= "false"  



if __name__ =="__main__":
    start_time= time.time() 
    train_loop(validate_epoch=True,
               pruning=False, 
                tokenizer_model=tokenizers.models["bert-cased"], 
                pretrain_embed=tokenizers.models["bert-cased"], 
                save_weights=False  
                ) 
    
    end_time= time.time() 
    print(f"Total Program Execution Time (min): {round((end_time-start_time)/60, 3)}")