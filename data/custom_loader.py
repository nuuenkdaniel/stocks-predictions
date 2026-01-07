'''' 
class of custom dataLoader for the data 
Given our dataset, this creates all the datapoints that can be used to make a DataLoader 
used for more efficient data processing  

- tokenization, create label 
- the getItem runs when we go through each batch during training (lazy evaluation)
'''
import torch 
from torch.utils.data import Dataset 
from transformers import AutoTokenizer 
from torch.nn.utils.rnn import pad_sequence

class SentimentDataset(Dataset): 
    def __init__(self, text, labels, model="bert-base-cased"):
        '''  
        @param text: all texts we want to be processed from the dataset 
        @param labels: corresponding labels of the dataset 
        '''
        
        self.tokenizer= AutoTokenizer.from_pretrained(model) 
        self.text= text 
        self.labels= labels 

        # confirm one label for each 
        assert(len(self.text) == len(self.labels))

    def __len__(self): 
        '''  
        return the length of the dataset (same as length of the text to be processed)
        '''
        return len(self.text) 


    def __getitem__(self, index):
        '''  
        return the tokenized (token id) of text in an index 
        '''

        token_ids= self.tokenizer.encode(self.text[index],
                                        add_special_tokens = True) 
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long), 
            "label": torch.tensor(self.labels[index], dtype=torch.long)
        }
    
    def get_vocab_size(self): 
        '''  
        return the vocab size of the tokenizer 
        '''
        return len(self.tokenizer)
    

def collate_fn(batch): 
    '''  
    dynamic padding for each batch (done when we are creating the dataLoader)
    @param batch: 
        - the batch (array) of data that's returned from SentimentDataset's getItem function 
        - batch_size number of (token_ids, label)
        - each token_id is the encoded version of the words 

    Find the longest sentence in the batch, and pad every other sentence to the longest length with PAD token 
    '''
    
    # sort from longest to shortest sentences in the batch 
    batch =sorted(batch, key= lambda x: len(x["token_ids"]), reverse=True)
    
    ids= [item["token_ids"] for item in batch]

    # Torch.Tensor of corresponding labels 
    labels= torch.stack([item["label"] for item in batch])

    # retrieve their pre-padding length for RNN to know where padding starts
    lengths= torch.tensor([len(item["token_ids"]) for item in batch])

    # pad token_ids with 0 
    padded= pad_sequence(ids, batch_first=True, padding_value=0)

    # what's returned in each batch 
    return padded, labels, lengths 
    
