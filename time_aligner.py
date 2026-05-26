import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class TimeAligner:
    def __init__(self, window_before: int = 1, window_after: int = 5):
        """
        :param window_before: Number of business days before the news to include.
        :param window_after: Number of business days after the news to include.
        """
        self.window_before = window_before
        self.window_after = window_after

    def align_and_vectorize(self, market_df: pd.DataFrame, news_df: pd.DataFrame):
        """
        Binds news events to market price windows using vector operations.
        
        market_df: columns [time, symbol, price]
        news_df: columns [time, symbol, sentiment_score]
        """
        if market_df.empty or news_df.empty:
            return pd.DataFrame()

        # 1. Prepare Market Data (Continuous Time Series)
        # Ensure time is index and sorted
        market_df = market_df.sort_values(['symbol', 'time'])
        
        # 2. Efficient Interpolation (Handling Holidays/Missing values)
        # Resample to daily frequency and forward fill to handle market closures
        # This is done per symbol to avoid cross-contamination
        market_df = market_df.set_index('time')
        
        def process_symbol_market(group):
            # Fill gaps (holidays) with last known price
            return group.resample('D').ffill()
        
        market_df = market_df.groupby('symbol', group_keys=False).apply(process_symbol_market)

        # 3. Vectorized Normalization (Z-Score)
        # Normalizing price globally per symbol to make it comparable
        market_df['norm_price'] = market_df.groupby('symbol')['price'].transform(
            lambda x: (x - x.mean()) / x.std()
        )

        # 4. Event-Based Binding (Merge Asof)
        # pd.merge_asof is much faster than for-loops for time alignment
        news_df = news_df.sort_values('time')
        
        # Merge news with the closest previous market state
        aligned = pd.merge_asof(
            news_df, 
            market_df[['symbol', 'norm_price']].reset_index(),
            on='time',
            by='symbol',
            direction='backward'
        )

        # 5. Window Extraction (Creating Integration Vectors)
        # Instead of looping, we use shift operations to get T-1, T+1, etc.
        # This assumes daily frequency due to the resample above.
        results = []
        for i in range(-self.window_before, self.window_after + 1):
            col_name = f'price_t{i:+d}' if i != 0 else 'price_t0'
            # Groupby symbol and shift norm_price
            aligned[col_name] = market_df.groupby('symbol')['norm_price'].shift(-i).reindex(aligned['time']).values
        
        return aligned

    def create_feature_vector(self, aligned_df: pd.DataFrame):
        """
        Converts aligned data into a clean NumPy matrix for ML.
        Rows: News Events | Columns: [Sentiment, T-1 Price, T0 Price, T+1 Price...]
        """
        feature_cols = ['sentiment_score'] + [c for c in aligned_df.columns if 'price_t' in c]
        
        # Drop rows with NaN in windows (usually at start/end of series)
        clean_df = aligned_df.dropna(subset=feature_cols)
        
        feature_matrix = clean_df[feature_cols].values
        return feature_matrix, clean_df
