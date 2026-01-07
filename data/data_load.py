import kagglehub
import os 
import pandas as pd  
import torch 
import sys 
from torch.utils.data import DataLoader, Dataset, random_split 
from .custom_loader import SentimentDataset, collate_fn 

def load_data(load_agreements=False): 
    '''
    return the DataFrame to the dataset which can be loaded and processed in diffrent file 

    the dataset contains data (text) and its labels 
    and the files of sentences which have different agreements (FinancialPhraseBank)
    '''
    # Download latest version, loaded into cache 
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")

    # load the downloaded dataset 
    data_path = os.path.join(path, 'all-data.csv')

    # directory of ['Sentences_66Agree.txt', 'Sentences_AllAgree.txt', 'Sentences_50Agree.txt', 'README.txt', 'License.txt', 'Sentences_75Agree.txt']
    agreement_path = os.path.join(path, "FinancialPhraseBank")
    
     # col 0: labels (neural, positive, negative)
    # col 1: news text 
    # 4846 x 2 
    # 2879 neutral, 1363 positive, 604 negative 
    # longest news: 315 words 
    # shortest news: 9 words 
    train_df= pd.read_csv(data_path, encoding='latin-1', header=None, names=["sentiment", "news"])  
    agreement_df=None 
    
    if (load_agreements): 
        agreement_df= pd.read_csv(agreement_path, encoding='latin-1', header=None, names=["sentiment", "news"]) 
        
    return train_df, agreement_df   
    


def process_dataset(df:pd.DataFrame,  
                 model = "bert-base-cased", 
                 load_agreements=False): 
    '''
    Step 2 
    Process the data to be trainable to get dataset 
    return the completed data in array format  

    @param load_agreement: whether to load and process data that has different agreement levels 
    '''
    header = df.columns.tolist()
    assert("sentiment" in header and "news" in header)

    # transform sentiment to numerical value 
    # neutral: 0, neg: 1, pos: 2 
    label_map= {"neutral":0, "negative":1, "positive":2}
    sentiment= (df["sentiment"].map(label_map))
    sentiment = sentiment.tolist() 
    news= (df["news"].tolist()) 

    # take non-torch.Tensor and create a dataset 
    dataset= SentimentDataset(news, sentiment, model) 
    vocab_size= dataset.get_vocab_size()
    print(f"Tokenizer model: {model} | Vocab Size: {vocab_size}")
    return dataset


def create_loaders(dataset, 
                batch_size, 
                train_size=0.8, 
                test_size=0.2, 
                dev_size=0, 
                seed=42,):
    '''  
    take the input dataset 
    create dataLoaders for training and testing loops

    ****based on train/dev/test split, not for k-fold cross validation*** 
    '''

    # split into train and test set size 
    train_size = int(train_size* len(dataset)) 
    test_size = len(dataset) - train_size 

    print(f"Training Set Size: {train_size} | Test Set Size: {test_size} News")

    # torch.utils.data.Subset object
    # calls __getItem__, so it's a tuple of (token_ids, label) in torch tensor  
    # same split with the same seed 
    train_set, test_set =  random_split(dataset, 
                                    [train_size, test_size], 
                                    generator=torch.Generator().manual_seed(seed) # reproducibility with manual seed
                                    )


    train_loader= DataLoader(train_set, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn, 
                            num_workers=2, 
                            shuffle=True   # shuffle training to prevent the model learning the order of the training data 
                            )
    test_loader= DataLoader(test_set,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False
                            )
    
    
    print(f"Train loader {len(train_loader)} batches | Test loader {len(test_loader)} batches")
    return train_loader, test_loader, train_size, test_size 




def retrieve_info(df:pd.DataFrame): 
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
    

if __name__ == "__main__": 
    pass 