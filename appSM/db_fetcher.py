import logging
from decouple import config
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


logger = logging.getLogger(__name__)


class ExternalDataNotFoundError(LookupError):
    """Lançada quando não há registros para os filtros informados."""


class ExternalDeviceNotFoundError(LookupError):
    """Lançada quando o dispositivo informado não existe no banco externo."""


class ExternalDataFetcher:
    """Leitura read-only do MySQL externo usando SQLAlchemy e pandas."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or config("EXTERNAL_MYSQL_URL") or config("EXTERNAL_DB_URL")
        if not self.database_url:
            raise ValueError("EXTERNAL_MYSQL_URL nao configurada")
        self._engine = create_engine(self.database_url, pool_pre_ping=True, future=True)

    def fetch_daily_history(self, sensor_id: str) -> pd.DataFrame:
        """Busca o histórico diário (45 dias incluindo hoje) agregado por sensor na tabela SensorData."""
        hoje = pd.Timestamp.now().normalize()
        inicio = hoje - pd.Timedelta(days=44) # 44 dias atrás + hoje = 45 dias
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
        return self._load_frame(query, params)
    
    def _fetch_dia_inicio_ciclo(self, dispositivo_id: str) -> int:
        """Busca o dia de fechamento na tabela Dispositivo e calcula o dia de início do ciclo."""
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
                
            if result and result[0] is not None:
                dia_fechamento = int(result[0])
                # Se fecha dia 13, o ciclo inicia dia 14
                dia_inicio = dia_fechamento
                return 1 if dia_inicio > 31 else dia_inicio
                
            return 1 # Fallback caso o registro não tenha a coluna preenchida
        except SQLAlchemyError as exc:
            logger.error("Erro ao buscar dia_fechamento_fatura para o dispositivo %s: %s", dispositivo_id, exc)
            return 1 # Fallback seguro

    def fetch_monthly_history(
        self,
        unidade_id: int,
        dispositivo_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Busca o histórico da RelatorioDiarioUnidade descobrindo o ciclo pelo dispositivo_id."""
        # Define o dia de início baseado no dispositivo ou assume o padrão 1
        dia_inicio_ciclo = 1
        if dispositivo_id is not None:
            dia_inicio_ciclo = self._fetch_dia_inicio_ciclo(dispositivo_id)

        hoje = pd.Timestamp.now().normalize()
        
        # Identifica o início do ciclo de faturamento atual
        if hoje.day >= dia_inicio_ciclo:
            inicio_ciclo_atual = hoje.replace(day=dia_inicio_ciclo)
        else:
            inicio_ciclo_atual = (hoje - pd.DateOffset(months=1)).replace(day=dia_inicio_ciclo)

        # Recua 11 meses para obter 12 ciclos completos de faturamento
        inicio_busca = inicio_ciclo_atual - pd.DateOffset(months=11)
        fim_exclusivo = hoje + pd.Timedelta(days=1)

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
        params = {"unidade_id": unidade_id, "inicio": inicio_busca.to_pydatetime(), "fim": fim_exclusivo.to_pydatetime()}
        daily_frame = self._load_frame(query, params)
        
        return self._aggregate_monthly(daily_frame, dia_inicio_ciclo)
    
    def _load_frame(self, query, params: dict) -> pd.DataFrame:
        try:
            df = pd.read_sql_query(query, self._engine, params=params, parse_dates=["Data"])
        except SQLAlchemyError as exc:
            logger.exception("Erro ao consultar banco externo: %s", exc)
            raise RuntimeError("Erro ao consultar banco externo") from exc

        if df.empty:
            raise ExternalDataNotFoundError("Nenhum registro de consumo encontrado para os filtros informados")

        df = df.copy()
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["Consumo"] = pd.to_numeric(df["Consumo"], errors="coerce")
        df = df.dropna(subset=["Data", "Consumo"]).sort_values("Data")

        if df.empty:
            raise ExternalDataNotFoundError("Nenhum registro de consumo encontrado após higienização")

        return df.set_index("Data")[["Consumo"]]
    
    

    @staticmethod
    def _aggregate_monthly(df: pd.DataFrame, dia_inicio_ciclo: int = 1) -> pd.DataFrame:
        working = df.copy()
        
        # Se o ciclo começa no dia 14, subtraímos 13 dias de todas as datas.
        # Isso faz com que o período de 14/04 a 13/05 seja "puxado" temporariamente para dentro do mês de Abril inteiro (01/04 a 30/04) no Pandas.
        offset_days = dia_inicio_ciclo - 1
        if offset_days > 0:
            working.index = working.index - pd.Timedelta(days=offset_days)
            
        # Agrupa pelo começo do mês (MS = Month Begin) somando os valores
        mensal = working.sort_index()[["Consumo"]].resample("MS").sum()
        
        if mensal.empty:
            raise ExternalDataNotFoundError("Nenhum registro de consumo consolidado.")
            
        # Devolvemos os dias retirados para que o índice volte a exibir a data real de início do ciclo (ex: 14/04)
        if offset_days > 0:
            mensal.index = mensal.index + pd.Timedelta(days=offset_days)

        return mensal

    def fetch_history_daily_report(
        self,
        unidade_id: int,
        data_inicio,
        data_fim,
        dias_historico: int = 45,
    ) -> pd.DataFrame:
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
        return self._load_frame(query, params)

    def fetch_dispositivo_dia_fechamento(self, dispositivo_id: str) -> int:
        """Busca o dia de fechamento/inicio do ciclo para o relatorio historico."""
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
            raise ExternalDeviceNotFoundError("Dispositivo nao encontrado")

        if result[0] is None:
            return 1

        dia_fechamento = int(result[0])
        if dia_fechamento < 1:
            return 1
        return min(dia_fechamento, 31)

    def fetch_history_monthly_report(
        self,
        unidade_id: int,
        inicio,
        fim,
        dia_fechamento_fatura: int,
    ) -> pd.DataFrame:
        """Busca dados diarios e agrega por ciclos mensais do relatorio historico."""
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
        daily_frame = self._load_frame(query, params)
        return self._aggregate_monthly(daily_frame, dia_fechamento_fatura)


def dataframe_para_historico(df: pd.DataFrame) -> dict[str, float]:
    """Converte um DataFrame temporal em dicionário compatível com o pipeline legado."""
    if df.empty:
        raise ExternalDataNotFoundError("Nenhum registro de consumo encontrado para os filtros informados")

    working = df.copy()
    historico = {
        data.strftime("%d/%m/%Y"): float(valor)
        for data, valor in working["Consumo"].items()
    }
    return historico
