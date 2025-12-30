import kagglehub
import os 
import pandas as pd  
import torch 
import sys 
from torch.utils.data import DataLoader, random_split 
from .custom_loader import SentimentDataset, collate_fn 

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
    

def data_process(batch_size, train_size=0.8, test_size=0.2, seed=42, load_agreements=False): 
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
    # longest news: 315 words 
    # shortest news: 9 words 
    df = pd.read_csv(data_path, encoding='latin-1', header=None, names=["sentiment", "news"]) 

    # transform sentiment to numerical value 
    # neutral: 0, neg: 1, pos: 2 
    label_map= {"neutral":0, "positive":2, "negative":1}
    sentiment= (df["sentiment"].map(label_map))
    sentiment = sentiment.tolist() 
    news= (df["news"].tolist()) 

    # take non-torch.Tensor and create a dataset 
    dataset= SentimentDataset(news, sentiment) 
    vocab_size= dataset.get_vocab_size() 

    # split into train and test set size 
    train_size = int(train_size* len(dataset)) 
    test_size = len(dataset) - train_size 

    # torch.utils.data.Subset object
    # calls __getItem__, so it's a tuple of (token_ids, label) in torch tensor  
    train_set, test_set =  random_split(dataset, 
                                    [train_size, test_size], 
                                    generator=torch.Generator().manual_seed(seed) # reproducibility with manual seed
                                    )


    train_loader= DataLoader(train_set, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn, 
                            num_workers=2, 
                            shuffle=False   # don't shuffle for training
                            )
    test_loader= DataLoader(test_set,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=True
                            )
    return train_loader, test_loader, vocab_size 


    


    ''' 
    # iterate thorugh dataset to get additional info 
    neutral= 0 
    positive=0 
    neg= 0 
    max_length=0 
    min_length= sys.maxsize 
    max_news=""

    # iterate thorugh each row as an Object 
    for row in df.itertuples(): 
        sent = row.sentiment 
        text= row.news 
        if (sent=="neutral"): 
            neutral +=1 
        elif (sent== "negative"):
            neg +=1 
        else: 
            positive+=1 
        if (len(text)>max_length):
            max_length= len(text)
            max_news= text 
        min_length= min(min_length, len(text))
    print(neutral)
    print(positive)
    print(neg)
    print(f"Longest news length: {max_length}")
    print(max_news)
    print(f"shortest news length: {min_length}")
    ''' 
    

if __name__ == "__main__": 
    data_process()