import requests
import pandas as pd
from datetime import datetime, timedelta
import os

class NewsCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query, start_date, end_date):
        """
        News API를 통해 특정 키워드의 뉴스를 수집합니다.
        """
        params = {
            'q': query,
            'from': start_date,
            'to': end_date,
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'language': 'en'
        }
        
        print(f"Fetching news for {query} from {start_date} to {end_date}...")
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return self._process_articles(articles, query)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def _process_articles(self, articles, ticker):
        processed = []
        for art in articles:
            processed.append({
                'ticker': ticker,
                'source': art['source']['name'],
                'title': art['title'],
                'description': art['description'],
                'content': art['content'],
                'url': art['url'],
                'published_at': art['publishedAt']
            })
        return pd.DataFrame(processed)

def save_to_csv(df, filename):
    os.makedirs('data/raw', exist_ok=True)
    path = os.path.join('data/raw', filename)
    df.to_csv(path, index=False)
    print(f"News data saved to {path}")

if __name__ == "__main__":
    # 실제 사용 시 환경 변수나 설정 파일에서 API KEY를 로드해야 합니다.
    API_KEY = "YOUR_NEWS_API_KEY" # 사용자가 직접 입력 필요
    collector = NewsCollector(API_KEY)
    
    ticker = "Apple" # 예시 키워드
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # API KEY가 설정되지 않은 경우를 대비한 가이드
    if API_KEY == "YOUR_NEWS_API_KEY":
        print("Please set your News API Key in the script.")
    else:
        news_df = collector.fetch_news(ticker, start_date, end_date)
        if news_df is not None and not news_df.empty:
            print(news_df.head())
            save_to_csv(news_df, f"{ticker}_news.csv")
