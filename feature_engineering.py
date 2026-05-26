import pandas as pd
import numpy as np
from src.data_ingestion.db_manager import DatabaseManager

class FeatureEngineer:
    def __init__(self, db_config):
        self.db_manager = DatabaseManager(db_config)

    def fetch_merged_data(self, ticker):
        """
        DB에서 주가 데이터와 뉴스 감성 요약 데이터를 가져와 결합합니다.
        """
        self.db_manager.connect()
        
        # 1. 주가 데이터 로드
        price_query = f"SELECT trade_date, open, high, low, close, volume FROM market_prices WHERE ticker = '{ticker}' ORDER BY trade_date"
        price_df = pd.read_sql(price_query, self.db_manager.conn)
        
        # 2. 일별 뉴스 요약 데이터 로드
        summary_query = f"SELECT summary_date, article_count, avg_sentiment FROM daily_news_summary WHERE ticker = '{ticker}' ORDER BY summary_date"
        summary_df = pd.read_sql(summary_query, self.db_manager.conn)
        
        self.db_manager.close()
        
        # 3. 데이터 결합 (Outer join 후 결측치 처리)
        merged_df = pd.merge(
            price_df, 
            summary_df, 
            left_on='trade_date', 
            right_on='summary_date', 
            how='left'
        )
        
        # 중복된 날짜 컬럼 삭제 (에러의 원인)
        if 'summary_date' in merged_df.columns:
            merged_df.drop(columns=['summary_date'], inplace=True)
            
        # 뉴스가 없는 날은 감성 점수 0, 기사 수 0으로 채움
        merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
        merged_df['article_count'] = merged_df['article_count'].fillna(0)
        
        return merged_df

    def create_features(self, df):
        """
        예측을 위한 추가 특성(Feature)들을 생성합니다.
        """
        df = df.sort_values('trade_date').copy()
        
        # 1. 기술적 지표: 수익률 (Return)
        df['return_1d'] = df['close'].pct_change()
        
        # 2. 감성 지수 시차 (Lag): 전날의 감성이 오늘 시가에 영향을 주도록 설정
        # (전날 장 마감 후 ~ 오늘 장 개시 전 뉴스가 중요하므로 1일 시차 적용)
        df['sentiment_lag1'] = df['avg_sentiment'].shift(1)
        
        # 3. 이동 평균 (Moving Average)
        df['price_ma5'] = df['close'].rolling(window=5).mean()
        df['sentiment_ma5'] = df['avg_sentiment'].rolling(window=5).mean()
        
        # 4. 타겟 변수 생성: 내일 시가 (Next Day Open) 또는 내일 시가 수익률
        # 우리는 '시가 예측'이 목적이므로 내일 시가를 타겟으로 잡음
        df['target_open'] = df['open'].shift(-1)
        
        # 결측치 제거
        df.dropna(inplace=True)
        
        return df

    def prepare_lstm_data(self, df, window_size=7):
        """
        LSTM 학습을 위한 슬라이딩 윈도우 데이터셋을 생성합니다.
        """
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sentiment_lag1', 'sentiment_ma5']
        X = []
        y = []
        
        data = df[feature_cols].values
        target = df['target_open'].values
        
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(target[i+window_size])
            
        return np.array(X), np.array(y)

if __name__ == "__main__":
    # 테스트용 설정
    DB_CONFIG = {
        "host": "localhost",
        "database": "capstone_db",
        "user": "postgres",
        "password": "yourpassword"
    }
    
    # engineer = FeatureEngineer(DB_CONFIG)
    # 실제 DB 데이터가 있을 때 실행:
    # df = engineer.fetch_merged_data("AAPL")
    # feat_df = engineer.create_features(df)
    # X, y = engineer.prepare_lstm_data(feat_df)
    # print(f"X shape: {X.shape}, y shape: {y.shape}")
