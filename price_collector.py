import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Yahoo Finance API를 통해 주가 데이터를 수집합니다.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        
        # 인덱스를 컬럼으로 변환 (Date -> trade_date)
        data.reset_index(inplace=True)
        data.rename(columns={
            'Date': 'trade_date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # 필요한 컬럼만 선택 및 Ticker 추가
        data['ticker'] = ticker
        return data[['ticker', 'trade_date', 'open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(df, filename):
    """
    데이터프레임을 CSV 파일로 저장합니다.
    """
    os.makedirs('data/raw', exist_ok=True)
    path = os.path.join('data/raw', filename)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    # 예시: 삼성전자(005930.KS) 또는 애플(AAPL) 데이터 수집
    ticker = "AAPL" 
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    stock_df = fetch_stock_data(ticker, start_date, end_date)
    if stock_df is not None:
        print(stock_df.head())
        save_to_csv(stock_df, f"{ticker}_prices.csv")
