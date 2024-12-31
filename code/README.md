
Some of packages in requirements.txt from my virtual environment are not needed. I recommend you only install packages when needed with 

```bash
pip install packackge_name
```
or 
```bash
python -m pip install packackge_name
```
on your terminal (MacOS or Linux) or PowerShell (Windows).
# Project 2: news sentiment analysis
- sentiment_analysis_alpaca_BERT: using pretrained transformer model FinBERT for financial news sentiment analysis
- nvda_alpaca_news.bz2: analysis results

There are also many other ways.


# Project 3: nn for trading

Backtesting with zipline may require quandl WIKI stock data. Will update that later.

- util.py: utility classes and functions
- gru_stock_pred.pth: trained GRU model ready to use
- nn_gru_backtesting_zipline: may need to modify the Path dependencies