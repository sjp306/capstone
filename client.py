import asyncpg
import logging
import pandas as pd
from config import DSN

logger = logging.getLogger(__name__)

class DBClient:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(DSN)
                logger.info("Database connection pool created.")
            except Exception as e:
                logger.error(f"Failed to create DB pool: {e}")
                raise

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

    async def get_latest_timestamp(self, table, symbol):
        """Returns the latest timestamp for a given symbol in a table."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT MAX(time) FROM {table} WHERE symbol = $1", symbol
            )
            return row['max'] if row else None

    async def insert_market_data(self, records):
        """Batch inserts market data records."""
        if not records:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO market_data (time, symbol, price, volume, metric_type)
                VALUES ($1, $2, $3, $4, $5)
                """,
                records
            )

    async def insert_news_data(self, records):
        """Batch inserts news data records with conflict handling."""
        if not records:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO news_data (time, symbol, title, content, sentiment_score, url)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (url, time) DO NOTHING
                """,
                records
            )

    async def fetch_market_df(self, symbols):
        """Fetches market data for symbols and returns a pandas DataFrame."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT time, symbol, price FROM market_data WHERE symbol = ANY($1) ORDER BY time",
                symbols
            )
            df = pd.DataFrame(rows, columns=['time', 'symbol', 'price'])
            df['price'] = df['price'].astype(float)
            return df

    async def fetch_news_df(self, symbols):
        """Fetches news data for symbols and returns a pandas DataFrame."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT time, symbol, sentiment_score, title FROM news_data WHERE symbol = ANY($1) ORDER BY time",
                symbols
            )
            df = pd.DataFrame(rows, columns=['time', 'symbol', 'sentiment_score', 'title'])
            df['sentiment_score'] = df['sentiment_score'].astype(float)
            return df

    # --- New Data Engineering Upgrades: Observability & Quality ---

    async def log_pipeline_start(self):
        """Logs the start of a pipeline run."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "INSERT INTO pipeline_logs (start_time, status) VALUES (NOW(), 'running') RETURNING id"
            )

    async def log_pipeline_end(self, log_id, status, market_count=0, news_count=0, error=None):
        """Logs the completion or failure of a pipeline run."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE pipeline_logs 
                SET end_time = NOW(), status = $2, market_records_count = $3, 
                    news_records_count = $4, error_message = $5
                WHERE id = $1
                """,
                log_id, status, market_count, news_count, error
            )

    def clean_market_data(self, records):
        """Data Quality Check: Filters out invalid market records."""
        cleaned = []
        now = datetime.now()
        for rec in records:
            # Check for: non-zero price, not in the future, valid volume
            if rec[2] > 0 and rec[0] <= now + timedelta(minutes=5):
                cleaned.append(rec)
        
        if len(records) != len(cleaned):
            logger.warning(f"Data Quality: Filtered out {len(records) - len(cleaned)} invalid market records.")
        return cleaned

    def clean_news_data(self, records):
        """Data Quality Check: Filters out invalid news records."""
        cleaned = []
        now = datetime.now()
        for rec in records:
            # Check for: non-empty title, valid URL, not in the future
            if rec[2] and rec[5] and rec[0] <= now + timedelta(minutes=5):
                cleaned.append(rec)
        
        if len(records) != len(cleaned):
            logger.warning(f"Data Quality: Filtered out {len(records) - len(cleaned)} invalid news records.")
        return cleaned
