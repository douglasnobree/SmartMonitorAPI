import pandas as pd
from typing import Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging
from zoneinfo import ZoneInfo
from decouple import config

from appSM.infrastructure.engine import get_engine
from appSM.infrastructure.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)
REPORT_TIME_ZONE = ZoneInfo(config("REPORT_TIME_ZONE", default="America/Fortaleza"))

def _local_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=REPORT_TIME_ZONE).tz_localize(None).normalize()

class ConsumoRepository:
    """Repositório read-only de dados de consumo do banco externo."""

    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def buscar_historico_diario(self, sensor_id: str) -> pd.DataFrame:
        """Busca o histórico diário (45 dias incluindo hoje) agregado por sensor."""
        from django.core.cache import cache
        hoje = _local_today()
        cache_key = f"consumo:diario:{sensor_id}:{hoje:%Y-%m-%d}"
        
        if (cached := cache.get(cache_key)) is not None:
            return cached

        inicio = hoje - pd.Timedelta(days=44)
        fim_exclusivo = hoje + pd.Timedelta(days=1)

        query = text(
            """
            SELECT DATE(data_leitura) AS Data, SUM(valor) AS Consumo
            FROM SensorData
            WHERE sensor_id = :sensor_id
              AND data_leitura >= :inicio
              AND data_leitura < :fim
            GROUP BY DATE(data_leitura)
            ORDER BY Data ASC
            """
        )
        params = {"sensor_id": sensor_id, "inicio": inicio.to_pydatetime(), "fim": fim_exclusivo.to_pydatetime()}
        df = self._carregar_dataframe(query, params)
        cache.set(cache_key, df, timeout=300)
        return df

    def buscar_historico_relatorio_diario(self, unidade_id: int, data_inicio, data_fim, dias_historico: int = 45) -> pd.DataFrame:
        """Busca dados diarios da unidade incluindo contexto anterior ao periodo solicitado."""
        inicio_busca = pd.Timestamp(data_inicio) - pd.Timedelta(days=dias_historico)
        fim_exclusivo = pd.Timestamp(data_fim) + pd.Timedelta(days=1)

        query = text(
            """
            SELECT data AS Data, valor_entrada AS Consumo
            FROM RelatorioDiarioUnidade
            WHERE id_unidade = :unidade_id
              AND data >= :inicio
              AND data < :fim
            ORDER BY data ASC
            """
        )
        params = {
            "unidade_id": unidade_id,
            "inicio": inicio_busca.to_pydatetime(),
            "fim": fim_exclusivo.to_pydatetime(),
        }
        return self._carregar_dataframe(query, params)
        
    def buscar_historico_mensal_bruto(self, unidade_id: int, inicio, fim) -> pd.DataFrame:
        """Busca dados diarios para agregação em ciclos mensais do relatorio historico."""
        query = text(
            """
            SELECT data AS Data, valor_entrada AS Consumo
            FROM RelatorioDiarioUnidade
            WHERE id_unidade = :unidade_id
              AND data >= :inicio
              AND data < :fim
            ORDER BY data ASC
            """
        )
        params = {
            "unidade_id": unidade_id,
            "inicio": pd.Timestamp(inicio).to_pydatetime(),
            "fim": pd.Timestamp(fim).to_pydatetime(),
        }
        return self._carregar_dataframe(query, params)

    def buscar_historico_mensal(self, unidade_id: int, dispositivo_id: Optional[str] = None) -> pd.DataFrame:
        """Busca histórico diário de uma unidade e agrega em ciclos mensais, usando dia de fechamento se dispositivo for informado."""
        hoje = _local_today()
        # Pega últimos 13 meses para garantir 12 meses fechados
        inicio = hoje - pd.DateOffset(months=13)
        fim = hoje + pd.Timedelta(days=1)
        
        df_bruto = self.buscar_historico_mensal_bruto(unidade_id, inicio, fim)
        
        dia_inicio_ciclo = 1
        if dispositivo_id:
            from appSM.infrastructure.dispositivo_repository import DispositivoRepository
            dia_inicio_ciclo = DispositivoRepository(self._engine).buscar_dia_fechamento(dispositivo_id)
            
        from appSM.domain.ciclo_faturamento import agregar_por_ciclo_mensal
        return agregar_por_ciclo_mensal(df_bruto, dia_inicio_ciclo)

    def _carregar_dataframe(self, query, params: dict) -> pd.DataFrame:
        try:
            df = pd.read_sql_query(query, self._engine, params=params, parse_dates=["Data"])
        except SQLAlchemyError as exc:
            logger.exception("Erro ao consultar banco externo: %s", exc)
            raise RuntimeError("Erro ao consultar banco externo") from exc

        if df.empty:
            raise DataNotFoundError("Nenhum registro de consumo encontrado para os filtros informados")

        df = df.copy()
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["Consumo"] = pd.to_numeric(df["Consumo"], errors="coerce")
        df = df.dropna(subset=["Data", "Consumo"]).sort_values("Data")

        if df.empty:
            raise DataNotFoundError("Nenhum registro de consumo encontrado após higienização")

        return df.set_index("Data")[["Consumo"]]
