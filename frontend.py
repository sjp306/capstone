import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import aiohttp
import asyncio
from datetime import datetime, timedelta

# --- [설정값: 본인의 NewsAPI 키를 입력하세요] ---
NEWS_API_KEY = "7a921f52ed5a4221aa1c8d52fbd8379e" 
NEWS_API_DELAY = 1.0
NEWS_API_CONCURRENCY = 3

# --- [1. 뉴스 수집 엔진] ---
class NewsClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.semaphore = asyncio.Semaphore(NEWS_API_CONCURRENCY)

    async def fetch_news(self, session, symbol: str, days: int = 7):
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        params = {
            'q': symbol,
            'from': from_date,
            'sortBy': 'relevance', # 관련도 높은 순
            'apiKey': self.api_key,
            'language': 'en',
            'pageSize': 5 # 시연용으로 5개만 수집
        }

        async with self.semaphore:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 429:
                        return []
                    data = await response.json()
                    if data.get('status') != 'ok':
                        return []
                    
                    articles = data.get('articles', [])
                    return [{
                        'time': art['publishedAt'],
                        'title': art['title'],
                        'content': art['description'] or art['content'],
                        'url': art['url']
                    } for art in articles]
            except Exception:
                return []

# --- [2. 감성 분석 엔진] ---
class FinancialSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze_text(self, text):
        if not text: return {"label": "neutral", "score": 0.0}
        return self.nlp(text[:512])[0]

    def convert_to_numeric_score(self, sentiment_result):
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']
        return score if label == 'positive' else -score if label == 'negative' else 0.0

# --- [3. 패턴 매칭 엔진] ---
class PatternMatcher:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def find_similar_patterns(self, target_series, historical_series, top_n=3):
        if len(target_series) < self.window_size or len(historical_series) < self.window_size:
            return []
        target_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-9)
        candidates = []
        for i in range(len(historical_series) - self.window_size + 1):
            window = historical_series[i : i + self.window_size]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-9)
            cos_sim = 1 - cosine(target_norm, window_norm)
            candidates.append({'start_idx': i, 'cos_sim': cos_sim, 'window': window_norm})
        
        candidates.sort(key=lambda x: x['cos_sim'], reverse=True)
        results = []
        for cand in candidates[:15]:
            dist, _ = fastdtw(target_norm, cand['window'], dist=2)
            results.append({'start_idx': cand['start_idx'], 'dtw_distance': float(dist)})
        results.sort(key=lambda x: x['dtw_distance'])
        return results[:top_n]

# --- [4. 앱 메인 설정] ---
st.set_page_config(page_title="자동 뉴스 리스크 분석기", layout="wide")

@st.cache_resource
def load_all_engines():
    return FinancialSentimentAnalyzer(), PatternMatcher(), NewsClient(NEWS_API_KEY)

sentiment_engine, pattern_engine, news_engine = load_all_engines()

# --- [5. 비동기 뉴스 수집 실행기] ---
async def run_news_collection(symbol):
    async with aiohttp.ClientSession() as session:
        return await news_engine.fetch_news(session, symbol)

# --- [6. UI 화면] ---
st.title("실시간 종목 리스크 자동 스캐너")
st.markdown("회사 이름을 넣으면 **최신 뉴스 수집 + 감성 분석 + 차트 패턴 매칭**을 한 번에 수행합니다.")

with st.sidebar:
    st.header("설정")
    company_name = st.text_input("회사명 (예: Apple, Nvidia, Tesla)", "Nvidia")
    analyze_btn = st.button("실시간 데이터 분석 시작")

if analyze_btn:
    ticker = yf.Search(company_name, max_results=1).quotes[0]['symbol'] if yf.Search(company_name).quotes else None
    
    if ticker:
        with st.spinner(f"'{company_name}'의 최신 뉴스와 주가 데이터를 가져오는 중..."):
            # 1. 뉴스 자동 수집 (비동기 실행)
            news_list = asyncio.run(run_news_collection(company_name))
            # 2. 주가 데이터 로드
            data = yf.download(ticker, period="2y")
            if isinstance(data.columns, pd.MultiIndex): data.columns = [c[0] for c in data.columns]

        if not news_list:
            st.warning("최근 수집된 뉴스가 없습니다. 직접 입력 모드를 사용하거나 키워드를 변경해 보세요.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"{company_name} ({ticker}) 주가 흐름")
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
                fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # 패턴 매칭
                target = data['Close'].iloc[-20:].values
                historical = data['Close'].iloc[:-20].values
                matches = pattern_engine.find_similar_patterns(target, historical)
                
                st.divider()
                st.subheader("과거 유사 패턴 매칭 결과")
                m_cols = st.columns(3)
                for idx, match in enumerate(matches):
                    match_date = data.index[match['start_idx']].strftime('%Y-%m-%d')
                    m_cols[idx].info(f"유사 시점: **{match_date}**\n\n패턴 거리: {match['dtw_distance']:.2f}")

            with col2:
                st.subheader("최신 뉴스 감성 분석")
                total_score = 0
                for i, news in enumerate(news_list):
                    res = sentiment_engine.analyze_text(news['title'])
                    score = sentiment_engine.convert_to_numeric_score(res)
                    total_score += score
                    
                    with st.expander(f"뉴스 {i+1}: {news['title'][:40]}..."):
                        st.write(f"**원본:** {news['title']}")
                        color = "green" if score > 0 else "red" if score < 0 else "gray"
                        st.markdown(f"**감성 진단:** :{color}[{res['label'].upper()}] ({score:.2f})")
                        st.caption(f"발행일: {news['time']}")
                        st.link_button("기사 원문 보기", news['url'])

                avg_score = total_score / len(news_list)
                st.divider()
                st.markdown(f"### 종합 리스크 지수: `{avg_score:.3f}`")
                if avg_score < -0.2:
                    st.error("현재 부정적인 뉴스가 지배적입니다. 리스크 관리가 필요합니다.")
                elif avg_score > 0.2:
                    st.success("긍정적인 뉴스가 많습니다. 시장 분위기가 호의적입니다.")
                else:
                    st.info("중립적인 상태입니다. 추가적인 시장 지표 확인이 필요합니다.")
    else:
        st.error("종목명을 찾을 수 없습니다.")

        # pip install streamlit yfinance pandas numpy plotly transformers torch scipy fastdtw aiohttp
        # streamlit run app.py