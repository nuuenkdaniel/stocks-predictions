import kagglehub
import os 
import pandas as pd 

def load_data(load_agreements=False): 
    '''
    return the path to the dataset which can be loaded and processed in diffrent file 

    the dataset contains data (text) and its labels 
    and the files of sentences which have different agreements (FinancialPhraseBank)
    '''
    # Download latest version, loaded into cache 
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")

    # load the downloaded dataset 
    data_path = os.path.join(path, 'all-data.csv')

    # directory of ['Sentences_66Agree.txt', 'Sentences_AllAgree.txt', 'Sentences_50Agree.txt', 'README.txt', 'License.txt', 'Sentences_75Agree.txt']
    agreement_path = os.path.join(path, "FinancialPhraseBank")
    
    if (load_agreements): 
        return data_path, agreement_path
    else:
        return data_path, None  

    

if __name__ == "__main__": 
    load_data() 