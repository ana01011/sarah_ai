import asyncio
from typing import List, Tuple
import asyncpg
import os

POSTGRES_URL = os.getenv("AGENT_MEMORY_POSTGRES_URL", "postgresql://demo_user:demo_pass@localhost:5432/agent_memory_demo")

class PersistentMemory:
    """
    Postgres-backed persistent memory for agents. Async, scalable, production-ready.
    """
    def __init__(self, dsn=POSTGRES_URL):
        self.dsn = dsn
        self._pool = None
        self._init_lock = asyncio.Lock()

    async def init(self):
        async with self._init_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
                async with self._pool.acquire() as conn:
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS agent_memory (
                            id SERIAL PRIMARY KEY,
                            agent_role TEXT NOT NULL,
                            sender TEXT NOT NULL,
                            message TEXT NOT NULL,
                            ts TIMESTAMP DEFAULT NOW()
                        );
                    ''')

    async def get(self, agent_role: str, limit: int = 20) -> List[Tuple[str, str]]:
        await self.init()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sender, message FROM agent_memory
                WHERE agent_role = $1
                ORDER BY ts DESC, id DESC
                LIMIT $2
                """, agent_role, limit
            )
            return [(r['sender'], r['message']) for r in reversed(rows)]

    async def add(self, agent_role: str, sender: str, message: str):
        await self.init()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memory (agent_role, sender, message) VALUES ($1, $2, $3)
                """, agent_role, sender, message
            )

    async def clear(self, agent_role: str):
        await self.init()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM agent_memory WHERE agent_role = $1
                """, agent_role
            )

persistent_memory = PersistentMemory()