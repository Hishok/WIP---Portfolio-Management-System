import yfinance as yf

def fetch_historical_prices(tickers, start_date, end_date):
    """
    Fetches historical adjusted close price data for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data
