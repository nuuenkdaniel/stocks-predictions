''' 
functions that implement feature pruning during data processing stage (pre-training)
'''

import json 
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from utils import create_path 


def chi_square_pruning(df:pd.DataFrame): 
    '''  
    feature pruning with chi-square test, keep the words with the highest score 
    '''
    # conver to document-term matrix
    # every sentence is aligned with all vocabs existed in the document 
    vectorizer = CountVectorizer(binary=True, min_df=2)  # remove all words that appear less than min_df headlines  
    X_counts = vectorizer.fit_transform(df["news"]) # transform into sparse matrix, where each column is an unique word from dictionary and each row is the sample sentence 
    y=df["sentiment"] 


    # initialize chi-square selector, trying to get rid of label independent words 
    #  chi-suqare tests each feature (word) with observed (number of times it occurs with corresponding label) and expected (50/50)
    # words with high score mean it's important (away from the 50/50 expectation) 
    # words with low score mean it's noisy (similar with the 50/50 expectation), it's independent of the label 
    selector = SelectKBest(score_func=chi2, k=2000) # keep top k number of words 

    # transofrm with noisy words thrown out 
    _ = selector.fit_transform(X_counts, y) 
   
    # get all kept words  
    all_words = list(vectorizer.vocabulary_.keys()) 
    selected_indices = selector.get_support(indices=True) 
    selected_features = [all_words[i] for i in selected_indices]

    return selected_features


def save_selected_features(selected_features):
    '''
    save the selected features from different pruning methods 
    ''' 

    print("---Selected Features/Dictionary Saved----")


def load_selected_features(path): 
    '''  
    load the selected dictionary 
    '''
    pass 


def prune_text(selected_features): 
    '''  
    given the input text, change words that didn't make into selected dictionary into [UNK] 
    '''