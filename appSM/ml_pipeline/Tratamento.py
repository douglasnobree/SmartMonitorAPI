"""
Interface abstrata para serviços de tratamento de dados.

Este módulo define a interface base que todos os serviços de processamento
de dados devem implementar no pipeline de Machine Learning.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


class Tratamento(ABC):
    """
    Classe abstrata base para serviços de tratamento de dados.
    
    Todos os serviços que processam dados (predição, classificação, análise)
    devem herdar desta classe e implementar o método processarDados.
    """

    @abstractmethod
    def processarDados(self, dados_request):
        """
        Processa os dados recebidos e retorna o resultado.
        
        Args:
            dados_request (dict): Dicionário com dados a serem processados.
                                 Formato esperado: {'DD/MM/YYYY': valor_float}
        
        Returns:
            dict ou float: Resultado do processamento (depende da implementação)
        
        Raises:
            Exception: Se houver erro no processamento dos dados
        """
        pass

    @staticmethod
    def _build_date_index(start: pd.Timestamp, end: pd.Timestamp, frequencia: str) -> Iterable[pd.Timestamp]:
        if frequencia == "mensal":
            datas = []
            atual = start
            while atual <= end:
                datas.append(atual)
                atual = atual + pd.DateOffset(months=1)
            return pd.DatetimeIndex(datas)

        return pd.date_range(start=start, end=end, freq="D")

    def _normalizar_historico(self, dados_request: Dict[str, float], frequencia: str) -> pd.DataFrame:
        df = pd.DataFrame({
            "Data": list(dados_request.keys()),
            "Consumo": list(dados_request.values())
        })

        df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
        df["Consumo"] = pd.to_numeric(df["Consumo"], errors="coerce")

        consumo_mediana = df["Consumo"].median()
        if pd.isna(consumo_mediana):
            consumo_mediana = 0.0

        invalidas = int(df["Data"].isna().sum())
        if invalidas > 0:
            logger.warning("Datas invalidas descartadas: %s", invalidas)

        df = df.dropna(subset=["Data"]).copy()
        if df.empty:
            raise ValueError("Nenhuma data valida encontrada no historico")

        contagens = df.groupby("Data")["Consumo"].size()
        datas_duplicadas = contagens[contagens > 1].index

        df = df.groupby("Data", as_index=False)["Consumo"].median()
        df["Consumo"] = df["Consumo"].fillna(consumo_mediana)

        if len(datas_duplicadas) > 0:
            df.loc[df["Data"].isin(datas_duplicadas), "Consumo"] = consumo_mediana
        df = df.sort_values("Data").reset_index(drop=True)

        index_datas = self._build_date_index(df["Data"].min(), df["Data"].max(), frequencia)
        df = (
            df.set_index("Data")
            .reindex(index_datas)
            .reset_index()
            .rename(columns={"index": "Data"})
        )

        df["Consumo"] = df["Consumo"].fillna(consumo_mediana)
        df["Data"] = df["Data"].dt.strftime("%d/%m/%Y")

        return df
