from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from core.config import settings

# asyncpg driver — pool_size = (CPU cores × 2) + 2
# On a 4-core App Service: pool_size=10, max_overflow=5
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,   # discard stale connections after idle
    pool_recycle=3600,    # recycle connections every hour
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
