''' 
functions needed to successfully deploy the model 
'''
# add parent root 
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_checkpoint, extract_bert_weight 
from data.tokenizer import init_tokenizer 
from model.rnn import LSTM 


global_hyperparams = None 

def load_model(checkpoint_addr:str):
    try: 
        checkpoint = load_checkpoint(checkpoint_addr)
    except Exception as e:
        print(f"Error Detected at loading checkpoint:\n\t{e}")
        return 
    assert("config" in checkpoint.keys())
    assert ("state_dict" in checkpoint.keys())
    hyperparams = checkpoint["config"] 

    # store hyperparams 
    global global_hyperparams
    
    global_hyperparams = hyperparams 

    print(f"Hyperparameters of model:\n\t{hyperparams}")

    tokenizer_model= hyperparams["tokenizer_model"]
    pretrain_embed = None 

    if ("pretrain_embed" in checkpoint.keys()): 
        pretrain_embed= checkpoint["pretrain_embed"]

    # make sure tokenizer and pretrained embedding weights are the same model 
    if(pretrain_embed):
        assert(tokenizer_model== pretrain_embed)
    
    embed_weights= extract_bert_weight(checkpoint["pretrain_embed"])
    
    model=hyperparams["model"] 
    if (model=="LSTM"):
        model = LSTM(n_layers=hyperparams["n_layers"],
                embed_dim=hyperparams["embed_dim"],
                hidden_dim=hyperparams["hidden_dim"],
                output_dim=hyperparams["output_dim"],
                vocab_size=hyperparams["vocab_size"],
                dropOut=hyperparams["dropOut"],
                embedding_weights=embed_weights)
    print(f"{model} Model Created")
    return model 
    

def create_tokenizer():
    '''  
    create tokenizer for future encoding and decoding 
    '''
    model = global_hyperparams["tokenizer_model"]
    tokenizer = init_tokenizer(model)
    print(f"{model} Tokenizer Created")
    return tokenizer 
