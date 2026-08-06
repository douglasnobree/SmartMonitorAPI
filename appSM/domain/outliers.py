from typing import Literal
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def tratar_outliers_iqr(
    df: pd.DataFrame,
    coluna: str = "Consumo",
    multiplicador: float = 1.5,
    substituicao: Literal["media", "mediana"] = "mediana",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Detecta outliers pelo método IQR e substitui pelo valor de referência.

    Args:
        multiplicador: 1.5 para classificação (mais sensível), 3.0 para predição (mais conservador)
        substituicao: "media" para classificação estatística, "mediana" para predição
    """
    if df.empty or coluna not in df.columns:
        return df, pd.Series([False] * len(df), index=df.index)

    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df, pd.Series([False] * len(df), index=df.index)

    limite_inferior = q1 - multiplicador * iqr
    limite_superior = q3 + multiplicador * iqr

    mascara_outliers = (
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    )

    total_outliers = int(mascara_outliers.sum())
    if total_outliers == 0:
        return df, mascara_outliers

    if substituicao == "media":
        valor_referencia = df.loc[~mascara_outliers, coluna].mean()
        if pd.isna(valor_referencia):
            valor_referencia = df[coluna].mean()
    else: # mediana
        valor_referencia = df.loc[~mascara_outliers, coluna].median()
        if pd.isna(valor_referencia):
            valor_referencia = df[coluna].median()

    df.loc[mascara_outliers, coluna] = valor_referencia

    logger.info(
        f"Outliers tratados: {total_outliers} valores substituídos "
        f"pela {substituicao} {valor_referencia:.4f}"
    )

    return df, mascara_outliers
