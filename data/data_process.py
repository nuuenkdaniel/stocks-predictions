import os 
import pandas as pd 
import numpy as np 
from data_load import load_data 

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
    df = pd.read_csv(data_path, encoding='latin-1', header=None) 

    


if __name__ == "__main__":
    data_process() 
