''' 
store hyperparameters and any settings for program execution 
'''
import model.tokenizers as tokenizers 


# here holds the hyperparameters for our model (global variable for our design)
# these parameters are mainly for model training (constructor params, training loop)
hyperparams= {
    "n_layers": 1, 
    "hidden_dim": 32,  # usually multiple of 2 
    "embed_dim": 32, 
    "output_dim": 3,    # we have 3 classes of labels to predict (neutral, pos, neg)
    "epochs": 8,        # number of times the model trains over the entire set 
    "batch_size": 64,   # batch size of each data training batch 
    "learning_rate": 0.001,  # learning rate of gradient descent 
    "dropOut": 0.5,     # reduce overfitting by randomly setting neurons to 0 weight 
    "weight_decay": 1e-5, 
    "l1_lambda": 0.000001, 
    "model": "LSTM", 
    "k_pruning": 4000,  # number of words to keep during chi-square pruning 

}

# validate at each epoch 
validate_epoch = True  

# use pruning while processing data 
pruning=False 

# tokenizer model 
tokenizer_model=tokenizers.models["bert-cased"]

# pretrain weights for embedding 
# MUST be the same as the tokenizer model used 
pretrain_embed=tokenizers.models["bert-cased"]

save_weights= False 


