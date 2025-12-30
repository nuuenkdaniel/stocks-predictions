# tokenizer for our dataset 
# a tokenizer is used to split a group of words into subwords (parts)
# a correct word is split into itself, but unusual/made-up words/mispelled words will be broken into differnt parts 
# a pretrained tokenizer is the best since it's trained to split words better than others 

''' 
This file may not be used in main code, but as tests 
'''

from transformers import AutoTokenizer 
import torch 

def init_tokenizer(model="bert-base-uncased"): 
    '''
    initialize the chosen pretrained tokenizer 
    '''

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # take the size of the tokenizer 
    #print(f"tokenizer dictionary size: {len(tokenizer)}") 

    return tokenizer 


def tokenize(text:str, tokenizer): 
    '''
    :param model: which pretrian model's tokenizer we can use from the HuggingFace library 
    :param text: the text to be tokenized 
    '''

    # this takes some computation (could be faster on gpu)
    # encode the input text into Ids (represent indices at a dictionary for example)
    # first breaks down the word into subwords, then change into indices in the model's dictionary 
    # ["input_ids", "token_type_ids, "attention_mask"]
    encoded = tokenizer(text)
    

    '''  
    print(f"Sample encoded: {encoded}")

    # convert tokens to words (see how tokenizer splits the subwords)
    # CLS: start of the sentence. SEP: separator to different sentences 
    tokens= tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    print(f"Splitted words: {tokens}")
    '''
    

    return encoded["input_ids"]
    

if __name__ == "__main__":
    tokenizer= init_tokenizer()
    tokenize("What is this? I want to go home!!!", tokenizer)