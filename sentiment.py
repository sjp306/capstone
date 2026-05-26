import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentProcessor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> float:
        """
        Analyzes text and returns the compound sentiment score.
        Score range: -1.0 (very negative) to 1.0 (very positive).
        """
        if not text:
            return 0.0
        scores = self.sia.polarity_scores(text)
        return scores['compound']

    def process_news_batch(self, news_articles):
        """Processes a batch of news articles and adds sentiment scores."""
        processed = []
        for article in news_articles:
            # Analyze title and content combined for better context
            text_to_analyze = f"{article['title']} {article['content'] or ''}"
            score = self.analyze(text_to_analyze)
            
            processed.append((
                article['time'],
                article['symbol'],
                article['title'],
                article['content'],
                score,
                article['url']
            ))
        return processed
