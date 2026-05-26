import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_ingestion.news_collector import NewsCollector
from src.data_ingestion.db_manager import DatabaseManager
from src.analysis.sentiment_analyzer import FinancialSentimentAnalyzer

class NewsAnalysisPipeline:
    def __init__(self, news_api_key, db_config):
        self.news_collector = NewsCollector(news_api_key)
        self.db_manager = DatabaseManager(db_config)
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        
    def run_daily_pipeline(self, ticker, query, days=1):
        """
        특정 티커에 대해 지정된 일수만큼의 뉴스를 수집, 분석하고 DB에 저장/집계합니다.
        """
        # 1. DB 연결
        self.db_manager.connect()
        
        # 2. 뉴스 수집
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        news_df = self.news_collector.fetch_news(query, start_date, end_date)
        
        if news_df is None or news_df.empty:
            print(f"No news found for {query}.")
            self.db_manager.close()
            return

        print(f"Collected {len(news_df)} articles. Starting sentiment analysis...")

        # 3. 감성 분석 및 실시간 DB 업데이트 (중간 결합)
        for _, row in news_df.iterrows():
            # 감성 분석 수행
            sentiment_res = self.sentiment_analyzer.analyze_text(row['title'])
            numeric_score = self.sentiment_analyzer.convert_to_numeric_score(sentiment_res)
            
            # 뉴스 발행 날짜 (Date 부분만 추출)
            pub_date = pd.to_datetime(row['published_at']).date()
            
            # DB 저장 및 중간 요약 업데이트
            # (참고: 실제 구현 시 news_articles 저장 로직도 db_manager에 추가 필요)
            self.db_manager.update_daily_summary(ticker, pub_date, numeric_score)
            
        print(f"Pipeline completed for {ticker}. Daily summaries updated.")
        self.db_manager.close()

if __name__ == "__main__":
    # 설정 정보
    NEWS_API_KEY = "7a921f52ed5a4221aa1c8d52fbd8379e" # 실제 키로 교체 필요
    DB_CONFIG = {
        "host": "localhost",
        "database": "capstone_db",
        "user": "postgres",
        "password": "yourpassword"
    }

    # 파이프라인 초기화 및 실행
    # 예: 애플(AAPL) 주식 관련 뉴스 분석
    pipeline = NewsAnalysisPipeline(NEWS_API_KEY, DB_CONFIG)
    
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Please set your News API Key in the script.")
    else:
        pipeline.run_daily_pipeline(ticker="AAPL", query="Apple Inc", days=3)
