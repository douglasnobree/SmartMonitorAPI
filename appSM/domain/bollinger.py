from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class EstrategiaBollinger(Protocol):
    """
    Contrato para qualquer algoritmo de cálculo de bandas estatísticas.
    Implemente este protocolo para trocar a estratégia sem modificar o use case.
    """
    def calcular(self, df: pd.DataFrame, janela: int) -> pd.DataFrame: ...

class RollingWindowBollinger:
    """
    Estratégia padrão: Bandas de Bollinger com rolling window de dias consecutivos.
    Calcula ±1σ, ±2σ, ±3σ em torno da média móvel.
    """
    def calcular(self, df: pd.DataFrame, janela: int) -> pd.DataFrame:
        df = df.copy()
        df["Média Móvel"] = df["Consumo"].rolling(window=janela, min_periods=1).mean()
        df["Desvio Padrão"] = df["Consumo"].rolling(window=janela, min_periods=1).std().fillna(0)
        df["Banda Inf 3"] = df["Média Móvel"] - 3 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Inf 2"] = df["Média Móvel"] - 2 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Inf 1"] = df["Média Móvel"] - 1 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Sup 1"] = df["Média Móvel"] + 1 * df["Desvio Padrão"]
        df["Banda Sup 2"] = df["Média Móvel"] + 2 * df["Desvio Padrão"]
        df["Banda Sup 3"] = df["Média Móvel"] + 3 * df["Desvio Padrão"]
        return df

estrategia_padrao = RollingWindowBollinger()
