import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_data(ticker:str, datapath:str=None) -> pd.DataFrame:
    if datapath is None:datapath = "data/raw_data.csv"
    sp500 = yf.Ticker(ticker)
    df = sp500.history(period="max"); df.to_csv(datapath)
    logging.info(f"dataframe saved at path {datapath}")
    return df

if __name__ == '__main__':
    df = get_data('^NSEI')
    print(df.head())