"""
Módulo de funções essenciais para tratamento e normalização de dados (Composição).

Substitui a antiga classe abstrata base Tratamento, seguindo o princípio
de favorecer Composição ao invés de Herança.
"""

import logging
from typing import Dict, Iterable, Union, Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_date_index(start: pd.Timestamp, end: pd.Timestamp, frequencia: str) -> Iterable[pd.Timestamp]:
    if frequencia == "mensal":
        datas = []
        atual = start
        while atual <= end:
            datas.append(atual)
            atual = atual + pd.DateOffset(months=1)
        return pd.DatetimeIndex(datas)

    return pd.date_range(start=start, end=end, freq="D")


def normalizar_historico(dados_request: Union[pd.DataFrame, pd.Series, Dict[Any, Any]], frequencia: str) -> pd.DataFrame:
    """
    Normaliza e padroniza o histórico de consumo, aceitando DataFrames, Series ou dicionários.
    Elimina a necessidade de conversões DataFrame->dict->DataFrame no pipeline.
    """
    if isinstance(dados_request, pd.DataFrame):
        if not dados_request.empty and "Consumo" in dados_request.columns:
            if isinstance(dados_request.index, pd.DatetimeIndex) and ("Data" not in dados_request.columns):
                df = dados_request.reset_index().copy()
                if "Data" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "Data"})
            else:
                df = dados_request.copy()
                if "Data" not in df.columns and len(df.columns) == 1:
                    df = df.reset_index().rename(columns={"index": "Data", 0: "Consumo"})
        elif dados_request.empty:
            df = pd.DataFrame(columns=["Data", "Consumo"])
        else:
            df = dados_request.reset_index().copy()
            if "Data" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Data"})
    elif isinstance(dados_request, pd.Series):
        df = dados_request.reset_index()
        df.columns = ["Data", "Consumo"]
    elif isinstance(dados_request, dict):
        df = pd.DataFrame({
            "Data": list(dados_request.keys()),
            "Consumo": list(dados_request.values())
        })
    else:
        raise TypeError("O formato dos dados não é suportado (deve ser DataFrame ou dict)")

    if not pd.api.types.is_datetime64_any_dtype(df["Data"]):
        df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
    else:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

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

    index_datas = build_date_index(df["Data"].min(), df["Data"].max(), frequencia)
    df = (
        df.set_index("Data")
        .reindex(index_datas)
        .reset_index()
        .rename(columns={"index": "Data"})
    )

    df["Consumo"] = df["Consumo"].fillna(consumo_mediana)
    df["Data"] = df["Data"].dt.strftime("%d/%m/%Y")

    return df
