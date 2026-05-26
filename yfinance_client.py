import yfinance as yf
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

async def fetch_yfinance_data(symbol: str, metric_type: str, start_date: datetime = None):
    """
    Fetches historical data from yfinance.
    Offloads blocking yfinance calls to a thread.
    """
    if not start_date:
        start_date = datetime.now() - timedelta(days=7)
    
    def _fetch():
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date)
        return df

    try:
        df = await asyncio.to_thread(_fetch)
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return []

        records = []
        for index, row in df.iterrows():
            records.append((
                index.to_pydatetime(),
                symbol,
                float(row['Close']),
                int(row['Volume']) if 'Volume' in row else 0,
                metric_type
            ))
        return records
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {symbol}: {e}")
        return []
