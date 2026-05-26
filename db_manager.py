import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import logging

class DatabaseManager:
    def __init__(self, db_config):
        self.config = db_config
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.config)
            print("Successfully connected to the database.")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def close(self):
        if self.conn:
            self.conn.close()

    def upsert_market_prices(self, df):
        """
        주가 데이터를 저장하고 중복 시 업데이트(Upsert) 합니다.
        """
        query = """
        INSERT INTO market_prices (ticker, trade_date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, trade_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """
        data = [tuple(x) for x in df.to_numpy()]
        with self.conn.cursor() as cur:
            execute_values(cur, query, data)
        self.conn.commit()
        print(f"Upserted {len(df)} price records.")

    def update_daily_summary(self, ticker, summary_date, sentiment_score):
        """
        뉴스 감성 점수가 들어오면 실시간으로 daily_news_summary를 업데이트합니다. (중간 결합)
        """
        query = """
        INSERT INTO daily_news_summary (ticker, summary_date, article_count, avg_sentiment)
        VALUES (%s, %s, 1, %s)
        ON CONFLICT (ticker, summary_date) DO UPDATE SET
            avg_sentiment = (daily_news_summary.avg_sentiment * daily_news_summary.article_count + EXCLUDED.avg_sentiment) / (daily_news_summary.article_count + 1),
            article_count = daily_news_summary.article_count + 1;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (ticker, summary_date, sentiment_score))
        self.conn.commit()

    def save_news_and_update_summary(self, ticker, news_data, sentiment_results):
        """
        뉴스 기사를 저장하고, 동시에 요약 테이블을 업데이트합니다.
        """
        # 1. 뉴스 저장 (news_articles)
        # 2. 감성 결과 저장 (news_sentiment)
        # 3. 요약 업데이트 (update_daily_summary 호출)
        pass

if __name__ == "__main__":
    # DB 연결 정보 (사용자 환경에 맞게 수정 필요)
    db_config = {
        "host": "localhost",
        "database": "capstone_db",
        "user": "postgres",
        "password": "yourpassword"
    }
    # manager = DatabaseManager(db_config)
    # manager.connect()
