from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
from app.core.config import get_settings
 
settings = get_settings()
 
# ── Async engine ──────────────────────────────────────────────────────────────
# NullPool: each request gets a fresh connection, returned immediately after.
# Appropriate for async FastAPI where connections are short-lived.
# For high-traffic production, switch to AsyncAdaptedQueuePool.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,       # Log SQL queries in debug mode
    pool_pre_ping=True,        # Verify connections before use
    pool_size=10,              # Max simultaneous connections
    max_overflow=20,           # Extra connections under load
)
 
# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # Keep objects accessible after commit
    autoflush=False,
    autocommit=False,
)
 
 
async def get_db() -> AsyncSession:
    """
    FastAPI dependency that provides a database session per request.
    Session is automatically closed after the request completes.
 
    Usage in endpoints:
        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Location))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()