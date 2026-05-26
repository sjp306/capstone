import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta

from config import NEWS_API_KEY
from db.client import DBClient
from collectors.yfinance_client import fetch_yfinance_data
from collectors.news_client import NewsClient
from processors.sentiment import SentimentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_pipeline(symbols):
    db = DBClient()
    await db.connect()
    
    # --- Observability: Log start ---
    log_id = await db.log_pipeline_start()
    market_count = 0
    news_count = 0
    
    try:
        news_client = NewsClient()
        sentiment_processor = SentimentProcessor()
        
        async with aiohttp.ClientSession() as session:
            # 1. Market Data Collection (yfinance)
            market_tasks = []
            for symbol in symbols:
                async with db.pool.acquire() as conn:
                    count = await conn.fetchval("SELECT count(*) FROM market_data WHERE symbol = $1", symbol)
                
                if count < 100:
                    start_date = datetime.now() - timedelta(days=365)
                    logger.info(f"Backfilling data for {symbol} (current count: {count})")
                else:
                    last_update = await db.get_latest_timestamp('market_data', symbol)
                    start_date = last_update if last_update else datetime.now() - timedelta(days=30)
                
                market_tasks.append(fetch_yfinance_data(symbol, 'equity', start_date))
            
            logger.info(f"Starting market data collection for {len(symbols)} symbols...")
            market_results = await asyncio.gather(*market_tasks)
            
            # Data Quality: Cleaning
            all_market_records = [rec for sublist in market_results for rec in sublist]
            cleaned_market_records = db.clean_market_data(all_market_records)
            
            if cleaned_market_records:
                await db.insert_market_data(cleaned_market_records)
                market_count = len(cleaned_market_records)
                logger.info(f"Inserted {market_count} clean market records.")

            # 2. News Data Collection & Sentiment Analysis
            news_tasks = []
            for symbol in symbols:
                news_tasks.append(news_client.fetch_news(session, symbol))
            
            logger.info(f"Starting news collection for {len(symbols)} symbols...")
            news_results = await asyncio.gather(*news_tasks)
            
            # Process and insert news
            total_processed_news = []
            for symbol_news in news_results:
                if symbol_news:
                    processed_news = sentiment_processor.process_news_batch(symbol_news)
                    total_processed_news.extend(processed_news)
            
            # Data Quality: Cleaning news
            cleaned_news_records = db.clean_news_data(total_processed_news)
            if cleaned_news_records:
                await db.insert_news_data(cleaned_news_records)
                news_count = len(cleaned_news_records)
                logger.info(f"Processed and inserted {news_count} clean news articles.")

        # --- Observability: Log success ---
        await db.log_pipeline_end(log_id, 'success', market_count, news_count)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # --- Observability: Log failure ---
        await db.log_pipeline_end(log_id, 'failed', market_count, news_count, error=str(e))
        raise
    finally:
        await db.close()

if __name__ == "__main__":
    target_symbols = ["AAPL", "GOOGL", "TSLA", "GC=F", "KRW=X"]
    
    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY not found in environment. Please check .env file.")
    else:
        try:
            asyncio.run(run_pipeline(target_symbols))
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
        except Exception as e:
            # Errors are already logged inside run_pipeline
            pass
