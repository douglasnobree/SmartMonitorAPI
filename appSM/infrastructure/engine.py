from decouple import config
from sqlalchemy import create_engine

_engine = None

def get_engine():
    """Module-level Lazy Engine — cria a conexão apenas quando necessário."""
    global _engine
    if _engine is None:
        url = config("EXTERNAL_MYSQL_URL", default=None) or config("EXTERNAL_DB_URL", default=None)
        if not url:
            raise ValueError("EXTERNAL_MYSQL_URL nao configurada")
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine
