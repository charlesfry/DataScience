
alpha_vantage_api_key = 'LA9GISTOLL4D2660' # <-- put your Alpha Vantage API key here as a string

# import essentials
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
import csv
import bs4 as bs
import os
import time

# get our S&P tickers
def save_sp500_tickers() :
    # request the webpage
    resp = requests.get(
        'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # parse the text
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    # get the first data cell in each row
    for row in table.findAll('tr')[1:] :
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())
    # write it to a new file. I chose to put it in my input folder
    with open('../input/sp500tickers.csv', 'w+', newline='') as f :
        write = csv.writer(f)
        write.writerow(tickers)
    # return sorted list of tickers
    tickers.sort()
    return tickers

# define function to get data
def get_alpha_vantage_daily_data(ticker, overwrite=False) :
    file_failures = 0
    time_series = TimeSeries(alpha_vantage_api_key, output_format='csv')
    if overwrite : print('Overwriting data...')
    # check if file already exists
    if (not os.path.exists(
            f'../input/StonkData/Alpha_Vantage/{ticker}.csv')) or overwrite is True:
        print(f'Getting {ticker}...')
        # if the file doesn't exist or we want to overwrite it, create it
        ts, meta = time_series.get_daily_adjusted(symbol=ticker,outputsize='full')
        with open(f'../input/StonkData/Alpha_Vantage/{ticker}.csv',
                  'w+',newline='') as fhand : # newline='' to prevent empty lines
            writer = csv.writer(fhand,dialect='excel')
            rowcount = 0
            for row in ts :
                writer.writerow(row)
                rowcount += 1
            # in case of API failure, delete file
            if rowcount < 10 :
                os.remove(f'../input/StonkData/Alpha_Vantage/{ticker}.csv')
                file_failures += 1
                print(f'Failed to obtain {ticker}')
            else :
                # can only get up to 5 responses per minute
                # so use time.sleep to prevent over-requesting
                time.sleep(12.01)
                print(f'    Obtained {ticker}')

    else : print(f'Already have {ticker}; moving on...')

    if file_failures > 0 : print(f'File requests denied: {file_failures}')



def compile_data() :
    print('\nCompiling Data...')

    # create main pandas df
    main_df = pd.DataFrame()
    fails = 0
    fail_tickers = []

    for count, ticker in enumerate(tickers):
        ticker = ticker.strip()

        print(ticker)
        file = f'../input/StonkData/Alpha_Vantage/{ticker}.csv'
        if os.path.exists(file) :
            df = pd.read_csv(file)
            try :
                df.rename(columns={'timestamp':'date'},inplace=True)
                df.set_index('date', inplace=True)
                df.drop(columns=[col for col in df.keys()
                                 if col not in ['adjusted_close']],
                        axis=1,inplace=True)
                df.rename(columns={'adjusted_close': ticker+'_adjusted_close'},
                          inplace=True)
                if main_df.empty : main_df = df
                else : main_df = main_df.join(df, how='outer')
            except :
                fails += 1
                fail_tickers.append(ticker)
            if count % 10 == 0 : print(count)
    print(main_df.tail())
    print(f'failed to join {fails} tables')
    print(fail_tickers)
    main_df = main_df.reindex(sorted(main_df.columns), axis=1)

    main_df.to_csv('../input/StonkData/Alpha_Vantage/joined_s&p_data.csv')
    return

# ------------------------
# now we run our functions

# get our tickers
tickers = save_sp500_tickers()

# ask if we should overwrite data
overwrite = input("Overwrite data? y/n\n").lower()
overwrite = overwrite in ['y','ye','yes']
if overwrite : print("Overwriting data...\n")

# get ticker data
for ticker in tickers :
    get_alpha_vantage_daily_data(ticker, overwrite)

compile_data()
