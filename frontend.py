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

NEWS_API_KEY = "7a921f52ed5a4221aa1c8d52fbd8379e" 
NEWS_API_DELAY = 1.0
NEWS_API_CONCURRENCY = 3

# --- [1. 뉴스 수집 엔진] ---
class NewsClient:
    def __init__(self, api_key):
        self.api_key = api_key # 외부에서 인자로 받은 값을 내부 변수에 저장
        self.base_url = "https://newsapi.org/v2/everything" # 뉴스데이터 url
        self.semaphore = asyncio.Semaphore(NEWS_API_CONCURRENCY)
# 비동기 동시성 제어
    async def fetch_news(self, session, symbol: str, days: int = 7):
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        params = {
            'q': symbol, # q(Quary) 심볼 단어가 들어간 뉴스를 찾음
            'from': from_date, # 시작 날짜, 이 날짜 이후의 뉴스만 필터링
            'sortBy': 'relevance', # 관련도 높은 순
            'apiKey': self.api_key, # API인증키
            'language': 'en', # 영어로 작성된 뉴스 수집
            'pageSize': 15 # 한번에 가져올 뉴스 15개 수집
        }

        async with self.semaphore: # 정해진 개수만큼 요청만 통과 나머지는 대기
            try: # 인터넷 연결 및 서버 에러 발생시 예외 처리
                async with session.get(self.base_url, params=params) as response:
                # 요청을 보내고 받음
                    if response.status == 429: # 429(너무 많은 요청)
                        return []
                    data = await response.json() # 서버가 보내준 json형식 뉴스 데이터를 딕셔너리 형태로 변환
                    if data.get('status') != 'ok': # NewsAPI가 성공했을때 응답
                        return [] # 실패했을때 빈 리스트 반환
                    
                    articles = data.get('articles', []) # 받아온 데이터에서 실제뉴스기사들이 모인 리스트 꺼냄
                    valid_articles = [] # 없으면 빈 리스트를 기본값으로 둔다
                    
                    for art in articles:
                        # 제목과 본문 내용이 모두 존재하는 기사만 필터링
                        title = art.get('title')
                        content = art.get('description') or art.get('content')
                        
                        if title and content: # 제목과 본문이 둘다 정상적으로 존재하는 기사만 수집
                            valid_articles.append({ # 유효한 기사 추가
                                'time': art['publishedAt'], # 발행시간
                                'title': title, # 제목
                                'content': content, # 본문
                                'url': art['url'] # 뉴스링크
                            })
                            
                        # 5개가 채워지면 루프를 종료하고 반환
                        if len(valid_articles) == 5:
                            break
                            
                    return valid_articles
            except Exception: # 에러 발생시 빈 리스트 반환
                return []

# --- [2. 감성 분석 엔진] ---
class FinancialSentimentAnalyzer:
    # 금융 뉴스 분석에 성능이 매우 뛰어난 사전 학습 Ai모델 FinBERT 사용
    def __init__(self, model_name="ProsusAI/finbert"):
        # 텍스트를 AI가 이해할 수 있도록 토큰 형태로 쪼갬
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 텍스트를 읽고 감정을 분류할 AI모델 본체를 불러옴
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        # Hugging Face 라이브러리의 pipeline 기능을 통해 토크나이저와 모델을 하나로 묵는 파이프라인 구축
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
# 텍스트 감성 분석
    def analyze_text(self, text):
        # 분석할 글이 비어 있으면 중립에 점수 0.0인 딕셔너리를 반환
        if not text: return {"label": "neutral", "score": 0.0}
        return self.nlp(text[:512])[0] # BERT 모델의 입력 제한(512글자)에 맞춰 잘라내고 분석
# 결과를 숫자로 변환
    def convert_to_numeric_score(self, sentiment_result):
        label = sentiment_result['label'].lower() # 결과 태그를 대소문자 문제 방지를 위해 모두 소문자로 바꿈
        score = sentiment_result['score'] # AI가 자신의 예측에 얼마나 확신하는지 나타내는 확률값(0.0~1.0사이)
# positive일때 확신점수 양수로 반환, negative일때 확신점수 음수로 반환, neutral일때 0.0 반환        
        return score if label == 'positive' else -score if label == 'negative' else 0.0

# --- [3. 패턴 매칭 엔진] ---
class PatternMatcher:
    def __init__(self, window_size: int = 20): # 차트 비교기간의 길이(창 크기) 20일
        self.window_size = window_size

    def find_similar_patterns(self, target_series, historical_series, top_n=3):
        # 비교대상인 현재차트나 과거차트의 데이터 개수가 최소기준 20일보다 작으면 분석불가능, 빈리스트 반환
        if len(target_series) < self.window_size or len(historical_series) < self.window_size:
            return []
        # 1. 스케일 맞추기 (정규화)
        # 오류 방지를 위해 금액 계급장을 떼고 오르고 내린 모양만 비교할 수 있도록 평균을 빼고 표준편차로 나눠 스케일 통일
        target_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-9)
        candidates = []
        # 2. 과거 2년 치 차트를 20일씩 슬라이딩하며 1차 후보군 필터링 (코사인 유사도)
        for i in range(len(historical_series) - self.window_size + 1):
            window = historical_series[i : i + self.window_size]
            # 잘라낸 과거 조각 차트도 똑같이 정규화
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-9)
            # 현재 차트와 코사인 유사도를 계산
            cos_sim = 1 - cosine(target_norm, window_norm)
            candidates.append({'start_idx': i, 'cos_sim': cos_sim, 'window': window_norm})
        # 3. 코사인 유사도가 높은 상위 15개를 뽑아 2차 정밀 검사 (DTW 알고리즘)
        # DTW는 시간축을 유연하게 늘리거나 줄여가며 흐름의 싱크로율을 찾아내는 고성능 알고리즘
        candidates.sort(key=lambda x: x['cos_sim'], reverse=True)
        results = []
        for cand in candidates[:15]:
            dist, _ = fastdtw(target_norm, cand['window'], dist=2)
            results.append({'start_idx': cand['start_idx'], 'dtw_distance': float(dist)})
        # 4. 차트의 거리가 가장 가까운 최종 3개 사례 선정
        results.sort(key=lambda x: x['dtw_distance'])
        return results[:top_n]

# --- [4. 앱 메인 설정] ---
st.set_page_config(page_title="자동 뉴스 리스크 분석기", layout="wide")
# 데코레이터, 서버가 켜질때 딱 한번만 메모리 공유를 통해 리소스 절약
@st.cache_resource
def load_all_engines():
    return FinancialSentimentAnalyzer(), PatternMatcher(), NewsClient(NEWS_API_KEY)
# 앱이 켜질 때 딱 한 번만 인공지능 모델과 엔진들을 로드함 (메모리 절약)
sentiment_engine, pattern_engine, news_engine = load_all_engines()

# --- [5. 비동기 뉴스 수집 실행기] ---
async def run_news_collection(symbol): # 비동기로 작동하는 함수
    async with aiohttp.ClientSession() as session: # 비동기로 웹요청을 보낼때 세션을 여는 코드
        return await news_engine.fetch_news(session, symbol)
# 비동기함수인 fetch_news가 실제 NewsAPI서버에 도착 하여 5개의 기사 수집때 까지 작업처리 지시
# --- [6. UI 화면] ---
st.title("실시간 종목 리스크 자동 스캐너")
st.markdown("회사 이름을 넣으면 **최신 뉴스 수집 + 감성 분석 + 차트 패턴 매칭**을 한 번에 수행합니다.")

with st.sidebar:
    st.header("설정")
    company_name = st.text_input("회사명 (예: Apple, Nvidia, Tesla)", "Nvidia")
    analyze_btn = st.button("실시간 데이터 분석 시작")

# 세션 상태(Session State) 초기화 세팅
if 'analyzed_data' not in st.session_state:
    st.session_state['analyzed_data'] = None

# 사용자가 '분석 시작' 버튼을 누르면 데이터를 새로 수집하고 세션에 저장
if analyze_btn:
    ticker = yf.Search(company_name, max_results=1).quotes[0]['symbol'] if yf.Search(company_name).quotes else None
    
    if ticker:
        with st.spinner(f"'{company_name}'의 최신 뉴스와 주가 데이터를 가져오는 중..."):
            # 1. 뉴스 자동 수집 (비동기 실행)
            news_list = asyncio.run(run_news_collection(company_name))
            # 2. 주가 데이터 로드
            data = yf.download(ticker, period="2y")
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = [c[0] for c in data.columns]

        if not news_list:
            st.warning("최근 수집된 뉴스가 없습니다. 키워드를 변경해 보세요.")
            st.session_state['analyzed_data'] = None
        else:
            # 패턴 매칭 연산 수행
            # 전체 기간 중 가장 최근 20일치 주가 데이터만 추출
            target = data['Close'].iloc[-20:].values
            # 처음부터 가장 최근 20일 전까지만 추출
            historical = data['Close'].iloc[:-20].values
            # 코사인 유사도로 필터링, 정밀도검사
            matches = pattern_engine.find_similar_patterns(target, historical)

            # 데이터 결과를 세션에 고정
            st.session_state['analyzed_data'] = {
                'company_name': company_name,
                'ticker': ticker,
                'news_list': news_list,
                'data': data,
                'matches': matches
            }
    else:
        st.error("종목명을 찾을 수 없습니다.")
        st.session_state['analyzed_data'] = None

# 세션에 데이터가 존재한경우 
if st.session_state['analyzed_data'] is not None:
    # 세션에서 안전하게 데이터 꺼내오기
    s_data = st.session_state['analyzed_data']
    c_name = s_data['company_name']
    tk = s_data['ticker']
    n_list = s_data['news_list']
    df_data = s_data['data']
    s_matches = s_data['matches']

    # 화면 레이아웃 좌우 분할 (2:1 비율)
    col1, col2 = st.columns([2, 1])

   
    # [왼쪽 열: 주가 차트 및 과거 패턴 매칭 기반 미래 예측]
   
    with col1:
        st.subheader(f"{c_name} ({tk}) 주가 흐름")
        fig = go.Figure(data=[go.Candlestick(
            x=df_data.index, 
            open=df_data['Open'], 
            high=df_data['High'], 
            low=df_data['Low'], 
            close=df_data['Close']
        )])
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        # 화면에 가로줄을 그어 분할
        st.divider()
        st.subheader("과거 유사 패턴 매칭 결과")
        
        if not s_matches:
            st.info("매칭된 과거 패턴이 없습니다.")
        else:
            # 1. 상단에 3개 요약 카드 고정 배치
            m_cols = st.columns(3)
            match_options = {}
            for idx, match in enumerate(s_matches):
                match_date = df_data.index[match['start_idx']].strftime('%Y-%m-%d')
                m_cols[idx].info(f"순위 {idx+1}위: **{match_date}**\n\n패턴 거리: {match['dtw_distance']:.2f}")
                match_options[f"{idx+1}위 사례 ({match_date} 기점)"] = match
            
            st.write("")
            
            # 2. 상세 내역 선택용 셀렉트박스
            selected_label = st.selectbox(
                "자세한 분석 정보를 보고 싶은 과거 사례를 선택하세요:", 
                list(match_options.keys())
            )
            
            # 3. 선택된 사례 상세 시각화 및 미래 예측
            if selected_label:
                selected_match = match_options[selected_label]
                start_i = selected_match['start_idx']
                
                # 과거 닮은 20일 구간 데이터
                matched_period_data = df_data.iloc[start_i : start_i + 20]
                
                # 과거 패턴이 시작된 지점으로부터 총 40일 뒤의 위치를 계산
                end_i = min(start_i + 40, len(df_data))
                # 과거 시점 당시 '사건 발생 이후 20일간'의 실제 주가 데이터
                past_future_data = df_data.iloc[start_i + 20 : end_i]
                # 과거차트의 가장 첫 번째 날짜를 가져와서 문자열로 반환
                past_start_date = matched_period_data.index[0].strftime('%Y-%m-%d')
                # 과거차트 조각의 가장 마지막(20일째) 날짜를 가져와 문자열로 변환
                past_end_date = matched_period_data.index[-1].strftime('%Y-%m-%d')
                
                # 미래 예측 로직 적용
                st.write("")
                st.markdown("###오늘 이후 미래 예측 시나리오")
                st.info(f"현재 발생한 리스크 패턴이 과거 **{past_end_date}** 당시에 발생했던 지정학적 리스크와 유사하게 흘러간다고 가정할 때, **오늘 이후 20거래일간의 미래 예측 시나리오**입니다.")
                
                # 1. 과거 사례의 '사건 이후 20일간의 변동률(수익률 배율)' 계산
                past_base_price = float(matched_period_data['Close'].iloc[-1]) # 과거 20일차 주가
                past_returns = []
                for p in past_future_data['Close'].values:
                    past_returns.append(p / past_base_price) # 당시 주가가 몇 배로 움직였는지 배율 저장
                
                # 2. 위에서 계산한 과거 배율을 '현재의 최신 주가'에 곱해서 미래 주가 시뮬레이션
                current_latest_price = float(df_data['Close'].iloc[-1]) # 오늘의 주가
                future_predictions = [current_latest_price * r for r in past_returns]
                
                # 3. 차트 그리기
                forecast_fig = go.Figure()
                
                # 트레이스 1: 최근 20일간의 실제 주가 흐름
                recent_20_days = df_data.iloc[-20:]
                forecast_fig.add_trace(go.Scatter( # 차트에 새로운 선이나 점 추가
                    x=recent_20_days.index, # x축 설정, 최근 20일간의 날짜
                    y=recent_20_days['Close'], # y축 설정, 최근 20일 동안의 주가
                    mode='lines+markers', # 차트의 형태, 꺽은선과 마커를 동시에 표시
                    name='최근 실제 주가 흐름 (~오늘까지)', # 차트의 라벨의 이름
                    line=dict(color='#1f77b4', width=3) # 실제 데이터임을 강조하도록 굵게 그림 
                ))
                
                # 트레이스 2: 오늘 이후 미래 20거래일 예측선
                if len(future_predictions) > 0:
                    # 미래 날짜 배열 생성
                    # 예측가격들이 어떤날짜의 가격인지 미래의 날짜들을 자동으로 계산해서 리스트로 만듬
                    future_dates = [df_data.index[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
                    
                    # 시각적으로 오늘 종가와 미래 예측선의 첫 점을 이어주기 위해 오늘 데이터 추가
                    forecast_dates_complete = [df_data.index[-1]] + future_dates
                    forecast_prices_complete = [current_latest_price] + future_predictions
                    
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast_dates_complete, # x축 설정, 미래의 날짜들
                        y=forecast_prices_complete, # y축 설정 미래 예측 가격들
                        mode='lines', # 선만 사용
                        name='실시간 기준 미래 예측 경로 (시나리오)', # 차트 우측 라벨에 뜨는 명찰이름
                        line=dict(color='#ff7f0e', width=4, dash='dash') # 주황색의 두꺼운 대시선 사용
                    ))
                # 최종 마무리 단계
                forecast_fig.update_layout(
                    template="plotly_dark", # 차트를 다크 모드로 바꿈
                    height=400, # 차트의 세로 높이를 400픽셀로 고정
                    margin=dict(l=20, r=20, t=20, b=20), # 차트외곽을 여백으로 20픽셀씩 줌
                    xaxis=dict(title="날짜 (Timeline)"), # x축 밑에 날짜 타임라인 표시
                    yaxis=dict(title="주가 ($)"), # y축 밑에 주가 단위 표시
                    # 그래프 설명창의 위치를 가로형태로 만듬
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                # Streamlit을 활용해 차트를 랜더링 하는 명령어 모든 화면 꽉 차게 가로길이를 자동으로 맞춤
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # 예측 스코어 브리핑
                if len(past_returns) > 0: # 과거 수익률 데이터가 존재 할때만 브리핑 실행
                    f_return = (past_returns[-1] - 1) * 100 # 과거 사례의 최종 수익률 변동폭
                    target_future_date = future_dates[-1].strftime('%Y-%m-%d')
                    
                    st.markdown(f"#### **시스템 미래 예측 브리핑**")
                    if f_return < -3:
                        st.warning(f" 과거 유사 사례 데이터 분석 결과, 리스크 발생 후 주가가 하락세를 겪었습니다. 이 패턴을 추종할 경우 **{target_future_date}** 경 주가는 현재보다 약 **{abs(f_return):.2f}% 하락**한 **${current_latest_price * past_returns[-1]:,.2f}**선까지 밀릴 리스크가 존재합니다.")
                    elif f_return > 3:
                        st.success(f" 과거 유사 사례 데이터 분석 결과, 지정학적 악재 충격 이후 빠르게 반등했습니다. 이 패턴을 추종할 경우 **{target_future_date}** 경 주가는 현재보다 약 **{f_return:.2f}% 상승**한 **${current_latest_price * past_returns[-1]:,.2f}**선까지 도달할 가능성이 예측됩니다.")
                    else:
                        st.info(f" 과거 유사 사례 데이터 분석 결과, 시장은 큰 변동 없이 박스권 횡보를 이어갔습니다. **{target_future_date}**까지 주가는 현재 가격 수준인 **${current_latest_price * past_returns[-1]:,.2f}** 내외에서 숨고르기를 할 것으로 보입니다.")

    
    # [오른쪽 열: 최신 뉴스 감성 분석]
    
    with col2: # 화면 반쪽 오른쪽 칸에 내용을 그림
        st.subheader("최신 뉴스 감성 분석")
        # 뉴스들의 감성 점수를 누적해서 더해나갈 공간 준비
        total_score = 0
        # 수집된 뉴스가 없는 경우
        if not n_list:
            st.info("수집된 뉴스가 없습니다.")
        else:
            # 긍정/부정/중립 딕셔너리를 받아옴
            for i, news in enumerate(n_list):
                res = sentiment_engine.analyze_text(news['title'])
                # 딕셔너리 결과를 계산이 가능하도록 score에 담고 이를 전체 점수에 누적
                score = sentiment_engine.convert_to_numeric_score(res)
                total_score += score
                # 페이지 늘어짐 방지를 위해 40글자만 잘라서 접이식으로 볼수있음
                with st.expander(f"뉴스 {i+1}: {news['title'][:40]}..."):
                    st.write(f"**원본:** {news['title']}")
                    # 점수가 양수면 초록색, 음수면 빨간색, 0이면 회색
                    color = "green" if score > 0 else "red" if score < 0 else "gray"
                    # Streamlit 마크다운 문법을 통해 AI가 판단한 라벨에 색상을 입힘
                    st.markdown(f"**감성 진단:** :{color}[{res['label'].upper()}] ({score:.2f})")
                    st.caption(f"발행일: {news['time']}")
                    st.link_button("기사 원문 보기", news['url'])
# 총점수를 뉴스개수로 나누어 평균 감성 점수(리스크 지수)를 산출
            avg_score = total_score / len(n_list)
            st.divider()
           
            st.markdown(f"### 종합 리스크 지수: `{avg_score:.3f}`")
             # -0.2미만 위험
            if avg_score < -0.2:
                st.error("현재 부정적인 뉴스가 지배적입니다. 리스크 관리가 필요합니다.")
            # 0.2 이상은 긍정
            elif avg_score > 0.2:
                st.success("긍정적인 뉴스가 많습니다. 시장 분위기가 호의적입니다.")
            # 그 외는 중립
            else:
                st.info("중립적인 상태입니다. 추가적인 시장 지표 확인이 필요합니다.")
