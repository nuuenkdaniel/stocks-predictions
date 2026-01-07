''' 
functions that implement feature pruning during data processing stage (pre-training)
'''

import json, os 
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from utils import get_timestamp, create_path 


def chi_square_pruning(df:pd.DataFrame, 
                       min_df=2, 
                       k=3000): 
    '''  
    feature pruning with chi-square test, keep the words with the highest score 
    '''
    header = df.columns.tolist()
    assert("sentiment" in header and "news" in header)

    # conver to document-term matrix
    # every sentence is aligned with all vocabs existed in the document 
    vectorizer = CountVectorizer(binary=True, min_df=min_df)  # remove all words that appear less than min_df headlines  
    X_counts = vectorizer.fit_transform(df["news"]) # transform into sparse matrix, where each column is an unique word from dictionary and each row is the sample sentence 
    y=df["sentiment"] 


    # initialize chi-square selector, trying to get rid of label independent words 
    #  chi-suqare tests each feature (word) with observed (number of times it occurs with corresponding label) and expected (50/50)
    # words with high score mean it's important (away from the 50/50 expectation) 
    # words with low score mean it's noisy (similar with the 50/50 expectation), it's independent of the label 
    selector = SelectKBest(score_func=chi2, k=k) # keep top k number of words 

    # transofrm with noisy words thrown out 
    _ = selector.fit_transform(X_counts, y) 
   
    # get all kept words  
    all_words = list(vectorizer.vocabulary_.keys()) 
    selected_indices = selector.get_support(indices=True) 
    selected_features = [all_words[i] for i in selected_indices]

    return set(selected_features) 


def prune_dataset(df:pd.DataFrame, selected_features):
    '''  
    apply pruning on the dataset 
    @return pruned dataset for creating dataset and dataloader 
    '''
    header = df.columns.tolist()
    assert("news" in header)

    def prune(word_list):
        return " ".join([word if word.lower() in selected_features else "[UNK]" for word in word_list.split()])    
    

    #  given the input text, change words that didn't make into selected dictionary into [UNK] 
    # modify in place
    df["news"]= df["news"].apply(prune)
    return

def save_selected_features(selected_features, path):
    '''
    save the selected features from different pruning methods 
    ''' 
    with open(path, "w") as f: 
        json.dump(list(selected_features), f)
    print("---Selected Features/Dictionary Saved----")
    return 


def load_selected_features(path): 
    '''  
    load the selected dictionary 
    '''
    with open(path,"r") as f:
        loaded = set(json.load(f))
    return loaded 


def handle_pruning(df:pd.DataFrame, k=3000, min_df=2): 
    selected_features=chi_square_pruning(df, min_df=min_df, k=k)
    prune_dataset(df, selected_features)

    # save and return path to the dictionary 
    path = create_path(folderName="saved_dictionaries", fileName=f"k={k} min_df={min_df} Chi-Square Prune.json")
    
    if (not os.path.exists(path)):
        save_selected_features(selected_features, path)
    return path 

