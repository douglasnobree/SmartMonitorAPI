from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging

from appSM.infrastructure.engine import get_engine
from appSM.infrastructure.exceptions import DeviceNotFoundError

logger = logging.getLogger(__name__)

class DispositivoRepository:
    """Repositório read-only de metadados de dispositivos."""

    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def buscar_dia_fechamento(self, dispositivo_id: str) -> int:
        """
        Busca o dia de fechamento da fatura do dispositivo.
        Lança DeviceNotFoundError se o dispositivo não existir.
        Retorna 1 como fallback se dia_fechamento_fatura for NULL ou inválido.
        """
        query = text(
            """
            SELECT dia_fechamento_fatura
            FROM Dispositivo
            WHERE id = :dispositivo_id
            """
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(query, {"dispositivo_id": dispositivo_id}).fetchone()
        except SQLAlchemyError as exc:
            logger.exception("Erro ao buscar dispositivo %s: %s", dispositivo_id, exc)
            raise RuntimeError("Erro ao consultar banco externo") from exc

        if result is None:
            raise DeviceNotFoundError("Dispositivo nao encontrado")

        if result[0] is None:
            return 1

        dia_fechamento = int(result[0])
        if dia_fechamento < 1:
            return 1
        return min(dia_fechamento, 31)
