import sys
import os
# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from src.data_ingestion.price_collector import fetch_stock_data
from src.data_ingestion.news_collector import NewsCollector
from src.data_ingestion.db_manager import DatabaseManager
from src.analysis.sentiment_analyzer import FinancialSentimentAnalyzer
from src.analysis.feature_engineering import FeatureEngineer

def run_full_system_test(ticker, query, news_api_key, db_config):
    """
    수집 -> 분석 -> 집계 -> 결합 전체 과정을 테스트합니다.
    """
    print(f"\n===== Starting Full System Test for {ticker} =====")
    
    # 1. 초기화
    db_manager = DatabaseManager(db_config)
    news_collector = NewsCollector(news_api_key)
    sentiment_analyzer = FinancialSentimentAnalyzer()
    feature_engineer = FeatureEngineer(db_config)
    
    try:
        # 2. DB 연결
        db_manager.connect()
        
        # 3. 주가 데이터 수집 및 저장
        print("\nStep 1: Fetching and Upserting Price Data...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        # 주가는 과거 패턴 학습을 위해 1년(365일)치를 가져옵니다.
        start_date_price = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_df = fetch_stock_data(ticker, start_date_price, end_date)
        if price_df is not None:
            db_manager.upsert_market_prices(price_df)
        
        # 4. 뉴스 데이터 수집, 분석 및 중간 집계
        print("\nStep 2: Fetching News and Updating Daily Summary...")
        # 뉴스는 무료 API 제한인 최근 30일치를 가져옵니다.
        start_date_news = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        news_df = news_collector.fetch_news(query, start_date_news, end_date)
        if news_df is not None and not news_df.empty:
            for _, row in news_df.iterrows():
                # 감성 분석
                sentiment_res = sentiment_analyzer.analyze_text(row['title'])
                score = sentiment_analyzer.convert_to_numeric_score(sentiment_res)
                # 요약 테이블 업데이트
                pub_date = pd.to_datetime(row['published_at']).date()
                db_manager.update_daily_summary(ticker, pub_date, score)
            print(f"Processed {len(news_df)} news articles.")

        # 5. 데이터 결합 및 특성 공학
        print("\nStep 3: Merging Data and Creating Features for AI...")
        merged_df = feature_engineer.fetch_merged_data(ticker)
        final_df = feature_engineer.create_features(merged_df)
        
        print("\n===== Test Result: Final Feature Set (Sample) =====")
        print(final_df.tail(5))
        
        # 6. LSTM 데이터셋 준비 확인
        X, y = feature_engineer.prepare_lstm_data(final_df)
        print(f"\nFinal Check: LSTM Input Shape {X.shape}, Target Shape {y.shape}")
        
    except Exception as e:
        print(f"\n[!] Test Failed: {e}")
        print("Tip: Make sure PostgreSQL is running and your API Key is valid.")
    finally:
        db_manager.close()
        print("\n===== Full System Test Finished =====")

if __name__ == "__main__":
    # 실행 전 필수 설정
    NEWS_API_KEY = "API_KEY"
    DB_CONFIG = {
        "host": "localhost",
        "database": "capstone_db",
        "user": "postgres",
        "password": "Password"
    }

    if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Error: Please provide a valid News API Key to run the full test.")
    else:
        run_full_system_test("AAPL", "Apple Inc", NEWS_API_KEY, DB_CONFIG)
