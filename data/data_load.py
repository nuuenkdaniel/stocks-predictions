import kagglehub
import os 
import pandas as pd 
import numpy as np 

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
    

def data_process(load_agreements=False): 
    '''
    Process the data to be trainable 
    return the completed data in array format  

    @param load_agreement: whether to load and process data that has different agreement levels 
    '''
    
    # load data 
    data_path, agreement_path = load_data(load_agreements)

    # col 0: labels (neural, positive, negative)
    # col 1: news text 
    # 4846 x 2 
    # 2879 neutral, 1363 positive, 604 negative 
    df = pd.read_csv(data_path, encoding='latin-1', header=None, names=["sentiment", "news"]) 

    ''' 
    # iterate thorugh dataset to get additional info 
    neutral= 0 
    positive=0 
    neg= 0 

    # iterate thorugh each row as an Object 
    for row in df.itertuples(): 
        sent = row.sentiment 
        if (sent=="neutral"): 
            neutral +=1 
        elif (sent== "negative"):
            neg +=1 
        else: 
            positive+=1 
    print(neutral)
    print(positive)
    print(neg)
    ''' 

    

if __name__ == "__main__": 
    data_process()