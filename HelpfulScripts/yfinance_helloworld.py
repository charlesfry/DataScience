# this module's imports
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # override yf imports with faster pdr

msft = yf.Ticker("MSFT")

# get stock info
print('\nType of msft.info:')
print(type(msft.info))
print()

# common yf commands

# get historical market data
hist = msft.history(period="max")

# show actions (dividends, splits)
print(msft.actions)

# show dividends
print(msft.dividends)

# show splits
print(msft.splits)

# show financials
print(msft.financials)
print(msft.quarterly_financials)

# show major holders
print(msft.major_holders)

# show institutional holders
print(msft.institutional_holders)

# show balance heet
print(msft.balance_sheet)
print(msft.quarterly_balance_sheet)

# show cashflow
print(msft.cashflow)
print(msft.quarterly_cashflow)

# show earnings
print(msft.earnings)
print(msft.quarterly_earnings)

# show sustainability
print(msft.sustainability)

# show analysts recommendations
print(msft.recommendations)

# show next event (earnings, etc)
print(msft.calendar)

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
print(msft.isin)

# show options expirations
print(msft.options)

# get option chain for specific expiration
#opt = msft.option_chain('YYYY-MM-DD') # is bugged
# data available via: opt.calls, opt.puts

data = pdr.get_data_yahoo("SPY", start="2020-01-01", end="2020-06-1")

print(data.tail())