

# Virtual environment and Python packages
It is highly recommended to create a virtual environment and do all the work in that environment. If you are not familair with it, RealPython has a good [primer](https://realpython.com/python-virtual-environments-a-primer/#create-it).

In Unix based system, create a virtual environment:
```bash
python3 -m venv venv/

```
Activate it: 
```bash
source venv/bin/activate
```
Install packages into it:

```bash
pip install packackge_name
```
or 
```bash
python -m pip install packackge_name
```
Some of packages in requirements.txt from my virtual environment are not needed. I recommend you only install packages when needed with 

# Project 2: news sentiment analysis 
To run the notebook, financial news through Alpaca api is needed. You might create your own api keys following [this tutorial](https://alpaca.markets/learn/connect-to-alpaca-api). Note you only need to do step 1 there to create your API keys. Or you could use my api keys that I will provide during the session.
Files:
- sentiment_analysis_alpaca_BERT: using pretrained transformer model FinBERT for financial news sentiment analysis
- nvda_alpaca_news.bz2: saved analysis results

# Project 3: nn for trading
To run the notebook, we need US equity data and zipline Python package.
### Get a QUANDL API Key
To download US equity data that we'll be using for zipline backtesting, [register](https://www.quandl.com/sign-up) for a personal Quandl account to obtain an API key. It will be displayed on your [profile](https://www.quandl.com/account/profile) page.

If you are on a UNIX-based system like Mac OSX, you may want to store the API key in an environment variable such as QUANDL_API_KEY, e.g. by adding `export QUANDL_API_KEY=<your_key>` to your `.bash_profile`.  

### Ingesting Zipline data
To install zipline, run
```bash
pip install zipline-reloaded
```
To run Zipline backtests, we need to `ingest` data. See the [Beginner Tutorial](https://zipline.ml4trading.io/beginner-tutorial.html) for more information. 

The image has been configured to store the data in a `.zipline` directory in the directory where you started the container (which should be the root folder of the starter code you've downloaded above). 

From the command prompt of the container shell, run
```bash
zipline ingest -b quandl
``` 
With that, you should be ready to run the notebook. Expect to see numerous messages as Zipline processes around 3,000 stock price series. Files:  
- util.py: utility classes and functions
- gru_stock_pred.pth: trained GRU model ready to use
- nn_gru_backtesting_zipline: may need to modify the Path dependencies

# Asian option pricing
C++ code for Asian option pricing with Monte Carlo and concurrency. Won't be covered. Only for your reference.