# stocks-predictions
- trying out AI models training (NLP) on stocks, market related dataset 

## Datasets 
- [sentiment analysis for financial market events 2025 (656kb)](https://www.kaggle.com/datasets/pratyushpuri/financial-news-market-events-dataset-2025)
- [Sentiment analysis for news (650kb)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- [Stock news sentiment analysis (9.43mb)](https://www.kaggle.com/datasets/avisheksood/stock-news-sentiment-analysismassive-dataset)
- [financial news sentiment database for market forecasting](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVW7SF)
- [News-setiment HuggingFace (292mb)](https://huggingface.co/datasets/sehyun66/News-sentiments)


## Steps 
- first try with Sentiment Analysis for news (650 kb), then use the stock news sentiment analysis, then combine both datasets. 
- compare results of both training using K-fold validation 



## Models 
- RNN (LSTM) model 
- Transformer model (smaller scale GPT2) 
- Finetune a pre-trained small scale lm from huggingFace 