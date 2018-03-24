# In[]: data scraping (S&P500 & Daily News)
import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import pandas_datareader

# Get tickers of companies that we want to scrape news
def get_tickers(): 
    tickers = []
    page_url = "https://www.slickcharts.com/sp500"
    page = requests.get(page_url)
    if page.status_code!=200: 
        page_url=None
    else:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select("div.panel-body table#example-1 tbody tr td div")
        for idx, div in enumerate(divs):
            ticker = None
            buf = div.select("input")[1]
            ticker = buf.attrs['value']
            tickers.append(ticker)
        tickers = tickers[0:100] # Select top 100 tickers by importance of components
    return tickers

# Update the format of tickers: sometimes there is a suffix representing the exchange after ticker 
def tickers_update(tickers): 
    updated = []
    for t in tickers:
        url = "http://www.reuters.com/finance/stocks/overview/" + t
        page = requests.get(url)
        if page.status_code!=200: 
            url=None
        else:
            soup = BeautifulSoup(page.content, 'html.parser')
            text = soup.select("div#sectionTitle h1")[0].get_text()
            updated_ticker = text[(text.find("(")+1):text.find(")")]
        updated.append(updated_ticker)
    return updated

# Select the top story news of the day
def get_news(tick, date): 
    page_url = "http://www.reuters.com/finance/stocks/company-news/" + tick + "?date=" +date
    page = requests.get(page_url)
    if page.status_code!=200: 
        page_url=None
    else:
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select("div#companyNews div div div.topStory h2")
        if len(divs)>0:
            return divs[0].get_text()
        else:
            return "NA"

# to get the daily movement of stock price: 1 for up and no change, 0 for down                       
def stock_movement(spx): 
    movement = []
    for i in range(1,len(spx)):
        if spx[i] >= spx[i-1]:
            movement.append(1)
        else:
            movement.append(0)
    return movement

# Extract dates of prices we got
def get_dates(spx): 
    date_list = spx.index
    datelist = []
    for i in range(len(date_list)):
        dt = str(spx.index[i]).split()[0]
        dt = datetime.datetime.strptime(dt, '%Y-%m-%d').strftime('%m%d%Y')
        datelist.append(dt)
    return datelist[1:]


# In[]

if __name__ == "__main__":
    # Download S&P 500 adjust close price from 2013-01-10 to 2017-10-07 
    spx = pandas_datareader.get_data_yahoo('^GSPC', start = datetime.datetime(2013, 01, 01),
                          end = datetime.datetime(2017, 10, 07))['Adj Close']
    
    # Get the movement of S&P 500
    movements = stock_movement(spx)
    
    # Get date of workday from 2013-01-01 to 2017-10-07  
    datelist = get_dates(spx) 
    movement_table = pd.DataFrame({'Date': datelist, 'Movement': movements})
    
    # Get ticker name of top 100 weight of companies
    tickers = get_tickers()
    tickers_updated = tickers_update(tickers)
    
    # Scraping adily news from Thomson Reuters
    news_data = []
    for d in datelist:
        print d
        daily = []
        daily.append(d)
        for t in tickers_updated:
            news = get_news(t, d)
            daily.append(news)
        news_data.append(daily)

    # Output csv 
    news_df = pd.DataFrame(news_data)   
    news_df.to_csv('news.csv', encoding='utf-8', index = False, header = False)
    movement_table.to_csv('movement.csv', encoding='utf-8', index = False, header = False)


# In[]: text cleaning
import csv
import nltk
import string  
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 


# Define a mapping between wordnet tags and POS tags as a function
def get_wordnet_pos(pos_tag):
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def data_clean(input_file,output_file):
    char = list(string.ascii_lowercase) # Get the list of alphabet
    #stop_words = set(stopwords.words('english'))
    #stop_words.update(('na', 'brief-', 'moves-', 'text -', 'text-', 'update ', 
    #                            'corrected-', 'refile-', 'wrapup', 'rpt-',
    #                            'us stocks-', 'on the move-'),char) # customize stopword and include na and alphabet in stopword
    # Sometimes the website has labels for news. They are not content of news so we remove lables such as BRIEF-, MOVES-.
    date = []
    news_cleaned = []
    with open(input_file,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            date.append(row[0]) # Get the date of workday
            doc = " ".join(row) # Join text of every row into a string
            doc = doc.lower() # Converts each document string into lower case
            pattern = r'[A-Za-z]+[A-Za-z\-\.]*'  # design pattern
            doc_without_num = nltk.regexp_tokenize(doc, pattern) # call NLTK's regular expression tokenization
            tokens=[token.strip(string.punctuation) for token in doc_without_num
                    if token.strip() not in string.punctuation] #remove stopwords and punctuation
            
            #lemmatization
            wordnet_lemmatizer = WordNetLemmatizer()
            tagged_tokens= nltk.pos_tag(tokens)

            #lemmatize every word in tagged_tokens
            le_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                     for (word, tag) in tagged_tokens # tagged_tokens is a list of tuples (word, tag)
                     if word not in string.punctuation] # remove punctuations
            #get lemmatized unique tokens as vocabulary
            tokens = set(le_words)
            news_cleaned.append(tokens)
            
    new_cleaned = pd.DataFrame(news_cleaned)
    new_cleaned.insert(0,'date',date)
    new_cleaned.to_csv(output_file, encoding='utf-8', index = False, header = False)


data_clean("/Users/Dido/midterm_C4 3news_2013-2017.csv","news_clean.csv")
