from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np

class FinancialSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        FinBERT 모델을 초기화합니다.
        기본적으로 ProsusAI/finbert 모델을 사용합니다.
        """
        print(f"Loading FinBERT model: {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # GPU 사용 가능 시 GPU 설정
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            print("GPU acceleration is enabled.")

    def analyze_text(self, text):
        """
        단일 텍스트에 대한 감성을 분석합니다.
        결과 예: {'label': 'positive', 'score': 0.95}
        """
        if not text or pd.isna(text):
            return {"label": "neutral", "score": 0.0}
        
        # 텍스트 길이 제한 (BERT 최대 토큰 수 512 고려)
        result = self.nlp(text[:512])[0]
        return result

    def analyze_batch(self, texts):
        """
        여러 텍스트를 배치 단위로 분석하여 속도를 향상시킵니다.
        """
        results = self.nlp(texts, truncation=True, padding=True)
        return results

    def convert_to_numeric_score(self, sentiment_result):
        """
        라벨을 -1.0(negative) ~ 1.0(positive) 사이의 숫자로 변환합니다.
        """
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:
            return 0.0

if __name__ == "__main__":
    # 테스트 코드
    analyzer = FinancialSentimentAnalyzer()
    
    test_titles = [
        "Apple reports record-breaking quarterly earnings, beating expectations.",
        "Stock market crashes as inflation fears mount among investors.",
        "The company announced a routine board meeting for next Tuesday."
    ]
    
    print("\n--- Sentiment Analysis Test ---")
    for title in test_titles:
        res = analyzer.analyze_text(title)
        num_score = analyzer.convert_to_numeric_score(res)
        print(f"Text: {title}")
        print(f"Result: {res['label']} (Confidence: {res['score']:.4f}), Numeric Score: {num_score:.4f}\n")
