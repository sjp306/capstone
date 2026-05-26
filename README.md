# Stock Analysis Data Pipeline

캡스톤 프로젝트를 위한 주식 데이터 수집 및 감성 분석 파이프라인입니다.

## 주요 기능
1.  **데이터 수집**: `yfinance`를 통한 정량 데이터(주가, 금리, 환율) 및 News API를 통한 비정형 뉴스 수집.
2.  **비동기 처리**: `asyncio`, `aiohttp`, `asyncpg`를 사용하여 높은 성능과 효율적인 I/O 처리.
3.  **로컬 캐싱**: TimescaleDB(PostgreSQL)를 연동하여 중복 수집 방지 및 증분 업데이트(Delta load).
4.  **감성 분석**: NLTK VADER를 사용하여 뉴스 텍스트를 -1.0 ~ 1.0 사이의 점수로 수치화.
5.  **Rate Limiting**: Semaphore와 Delay 전략을 통해 API 요청 제한 준수.

## 설치 및 설정

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터베이스 설정
PostgreSQL에 TimescaleDB 확장이 설치되어 있어야 합니다.
`db/schema.sql` 파일을 실행하여 테이블과 하이퍼테이블을 생성하세요.

### 3. 환경 변수 설정
`.env.example` 파일을 `.env`로 복사하고 실제 정보를 입력하세요.
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
NEWS_API_KEY=your_news_api_key
```

## 실행 방법
```bash
python main.py
```

## 프로젝트 구조
- `main.py`: 파이프라인 통합 실행
- `config.py`: 환경 변수 및 설정 관리
- `db/`: 데이터베이스 클라이언트 및 스키마
- `collectors/`: yfinance 및 News API 수집 모듈
- `processors/`: 감성 분석 처리 모듈
