import asyncio
from db.client import DBClient

async def check_data():
    db = DBClient()
    await db.connect()
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("SELECT symbol, count(*) FROM market_data GROUP BY symbol;")
        for row in rows:
            print(f"Symbol: {row['symbol']}, Count: {row['count']}")
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_data())
