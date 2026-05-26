from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
import pandas as pd
import json
import redis.asyncio as redis
from datetime import datetime, timedelta

from db.client import DBClient
from processors.pattern_matcher import PatternMatcher

app = FastAPI(title="Stock Pattern Analysis API")

# 1. Performance: GZip Compression for large datasets
app.add_middleware(GZipMiddleware, minimum_size=1000)

db_client = DBClient()
matcher = PatternMatcher(window_size=20)
redis_client: Optional[redis.Redis] = None

class SimilarityRequest(BaseModel):
    symbol: str
    target_window_days: int = 20
    top_n: int = 10

# In-memory storage for background task results
task_status = {}

@app.on_event("startup")
async def startup():
    global redis_client
    # Try connecting to DB
    try:
        await db_client.connect()
        print("Connected to Database")
    except Exception as e:
        print(f"Database connection failed: {e}")

    # 2. Performance: Initialize Redis Caching
    try:
        redis_client = redis.from_url("redis://localhost", decode_responses=True)
        await redis_client.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Redis connection failed: {e}. Caching disabled.")
        redis_client = None

@app.on_event("shutdown")
async def shutdown():
    await db_client.close()
    if redis_client:
        await redis_client.close()

async def run_analysis_task(task_id: str, symbol: str, window_days: int, top_n: int):
    """Background task for heavy DTW computation."""
    try:
        task_status[task_id] = {"status": "processing"}
        
        # Fetch data
        df = await db_client.fetch_market_df([symbol])
        if df.empty:
            task_status[task_id] = {"status": "failed", "error": f"No data found for {symbol}"}
            return

        df = df.sort_values('time')
        prices = df['price'].values
        times = df['time'].values

        if len(prices) < window_days * 2:
            task_status[task_id] = {"status": "failed", "error": "Insufficient historical data"}
            return

        target_pattern = prices[-window_days:]
        historical_data = prices[:-window_days]
        historical_times = times[:-window_days]

        # Heavy DTW Matching
        matches = matcher.find_similar_patterns(target_pattern, historical_data, top_n=top_n)

        # Format Results
        results = []
        for match in matches:
            idx = match['start_idx']
            # Convert numpy.datetime64 to python datetime for isoformat
            start_date = pd.to_datetime(historical_times[idx]).to_pydatetime()
            results.append({
                "start_date": start_date.isoformat(),
                "dtw_distance": match['dtw_distance'],
                "similarity": match['cos_sim'],
                "values": historical_data[idx : idx + window_days].tolist()
            })

        task_status[task_id] = {
            "status": "completed",
            "symbol": symbol,
            "results": results
        }
        
        # Cache Result in Redis (Expire in 1 hour)
        if redis_client:
            cache_key = f"similarity:{symbol}:{window_days}:{top_n}"
            await redis_client.set(cache_key, json.dumps(task_status[task_id]), ex=3600)

    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/similarity-search")
async def request_search(request: SimilarityRequest, background_tasks: BackgroundTasks):
    """Entry point for similarity search."""
    
    # Check Redis Cache First
    if redis_client:
        try:
            cache_key = f"similarity:{request.symbol}:{request.target_window_days}:{request.top_n}"
            cached = await redis_client.get(cache_key)
            if cached:
                print("Cache hit!")
                return json.loads(cached)
        except Exception as e:
            print(f"Redis cache error: {e}")

    # Background Task Execution
    task_id = f"task_{request.symbol}_{int(datetime.now().timestamp())}"
    background_tasks.add_task(
        run_analysis_task, 
        task_id, 
        request.symbol, 
        request.target_window_days, 
        request.top_n
    )
    
    return {"task_id": task_id, "status": "queued", "message": "Heavy analysis started in background."}

@app.get("/api/v1/tasks/{task_id}")
async def get_task_result(
    task_id: str, 
    page: int = Query(1, ge=1), 
    page_size: int = Query(5, ge=1, le=20)
):
    """Returns partial results of a completed task."""
    task = task_status.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "completed":
        return task

    # Implement Pagination
    all_results = task["results"]
    total = len(all_results)
    start = (page - 1) * page_size
    end = start + page_size
    
    paginated_results = all_results[start:end]
    
    return {
        "status": "completed",
        "symbol": task["symbol"],
        "total_results": total,
        "page": page,
        "page_size": page_size,
        "has_next": end < total,
        "results": paginated_results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
