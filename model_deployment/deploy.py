''' 
contain functions to deploy and run the model in the background 
'''
from model_utils import load_model, create_tokenizer 

default_checkpoint= "../checkpoints/8 Epochs LSTM Test Acc=0.743 Train Acc= 0.983 | 01-05_22-06.pt" 


def deploy(checkpoint_addr=default_checkpoint):
    '''  
    when called, starts user interfae and launch the model 

    steps: 
        - create model and make it runnable 
        - give user prompt to enter text 
        - process input 
        - run model to get outputs 
        - return to user 
    ''' 
    # create model 
    model = load_model(checkpoint_addr) 
    # create tokenizer 
    tokenizer= create_tokenizer()
    
    # loop of interaction 
    
    
    



if __name__ == "__main__": 
    deploy() 