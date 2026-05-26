import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from config import NEWS_API_KEY, NEWS_API_CONCURRENCY, NEWS_API_DELAY

logger = logging.getLogger(__name__)

class NewsClient:
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        self.semaphore = asyncio.Semaphore(NEWS_API_CONCURRENCY)

    async def fetch_news(self, session, symbol: str, days: int = 30):
        """
        Fetches news for a symbol using News API.
        Implements rate limiting via semaphore and delay.
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        params = {
            'q': symbol,
            'from': from_date,
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'language': 'en'
        }

        async with self.semaphore:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 429:
                        logger.warning("Rate limit hit for News API. Sleeping...")
                        await asyncio.sleep(60) # Back off
                        return []
                    
                    data = await response.json()
                    if data.get('status') != 'ok':
                        logger.error(f"News API error: {data.get('message')}")
                        return []

                    articles = data.get('articles', [])
                    results = []
                    for art in articles:
                        results.append({
                            'time': datetime.fromisoformat(art['publishedAt'].replace('Z', '+00:00')),
                            'symbol': symbol,
                            'title': art['title'],
                            'content': art['description'] or art['content'],
                            'url': art['url']
                        })
                    
                    # Respect delay between requests
                    await asyncio.sleep(NEWS_API_DELAY)
                    return results

            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                return []
