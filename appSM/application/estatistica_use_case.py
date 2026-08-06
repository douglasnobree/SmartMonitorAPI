from appSM.domain.tratamento import normalizar_historico
from appSM.domain.outliers import tratar_outliers_iqr
from appSM.domain.bollinger import estrategia_padrao, EstrategiaBollinger
from appSM.domain.classificador import classificar_serie, classificar_por_faixa
from appSM.infrastructure.consumo_repository import ConsumoRepository
from appSM.infrastructure.exceptions import DataNotFoundError
from appSM.application.exceptions import ConsumoNaoEncontrado
import pandas as pd


class EstatisticaUseCase:
    """
    Casos de uso de análise estatística de consumo (Bollinger).
    Cada método é um caso de uso independente — a view chama exatamente um.
    """

    @staticmethod
    def diario(sensor_id: str, estrategia: EstrategiaBollinger = None) -> dict:
        """
        Caso de uso: análise estatística diária por sensor.
        Parâmetro estrategia permite injetar algoritmo alternativo de bandas (ex: por dia da semana).
        """
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline(df, janela=30, frequencia="diaria", estrategia=estrategia)

    @staticmethod
    def mensal(unidade_id: int, dispositivo_id: str = None, estrategia: EstrategiaBollinger = None) -> dict:
        """Caso de uso: análise estatística mensal por unidade."""
        try:
            df = ConsumoRepository().buscar_historico_mensal(unidade_id, dispositivo_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline(df, janela=12, frequencia="mensal", estrategia=estrategia)

    @staticmethod
    def dados_completos(sensor_id: str) -> list:
        """Caso de uso: série completa de bandas para visualização gráfica."""
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline_completo(df, janela=30, frequencia="diaria")


def _executar_pipeline(df: pd.DataFrame, janela: int, frequencia: str, estrategia: EstrategiaBollinger = None) -> dict:
    """Pipeline interno de análise estatística — sem estado, sem efeitos colaterais."""
    _estrategia = estrategia or estrategia_padrao

    df = normalizar_historico(df, frequencia=frequencia)
    df_original = df.copy()
    df, mascara_outliers = tratar_outliers_iqr(df, multiplicador=1.5, substituicao="media")
    df = _estrategia.calcular(df, janela=janela)
    df["Classificação"] = classificar_serie(df)

    # Lógica de outlier no último ponto: classificar pelo valor original
    last_idx = df.index[-1]
    if bool(mascara_outliers.loc[last_idx]):
        consumo_original = float(df_original.loc[last_idx, "Consumo"])
        row = df.loc[last_idx]
        df.loc[last_idx, "Classificação"] = classificar_por_faixa(
            consumo_original, 
            row["Banda Sup 2"], row["Banda Sup 1"], 
            row["Banda Inf 1"], row["Banda Inf 2"]
        )
        df.loc[last_idx, "Consumo"] = consumo_original

    df = _preencher_nulos(df)
    ultima_linha = df.iloc[-1]
    return {
        "Data": ultima_linha["Data"],
        "Consumo": ultima_linha["Consumo"],
        "Classificação": int(ultima_linha["Classificação"]) if pd.notna(ultima_linha["Classificação"]) else None,
    }


def _executar_pipeline_completo(df: pd.DataFrame, janela: int, frequencia: str, estrategia: EstrategiaBollinger = None) -> list:
    """Pipeline para obter toda a série histórica com bandas e classificações."""
    _estrategia = estrategia or estrategia_padrao

    df = normalizar_historico(df, frequencia=frequencia)
    df_original = df.copy()
    df, mascara_outliers = tratar_outliers_iqr(df, multiplicador=1.5, substituicao="media")
    df = _estrategia.calcular(df, janela=janela)
    df["Classificação"] = classificar_serie(df)
    
    # Restaurar o valor original onde houve outliers para a visualização
    df.loc[mascara_outliers, "Consumo"] = df_original.loc[mascara_outliers, "Consumo"]
    
    df = _preencher_nulos(df)
    
    registros = []
    for idx, row in df.iterrows():
        registros.append({
            "Data": row["Data"],
            "Consumo": round(row["Consumo"], 2),
            "Média Móvel": round(row["Média Móvel"], 2) if not pd.isna(row["Média Móvel"]) else None,
            "Desvio Padrão": round(row["Desvio Padrão"], 2) if not pd.isna(row["Desvio Padrão"]) else None,
            "Banda Inf 3": round(row["Banda Inf 3"], 2) if not pd.isna(row["Banda Inf 3"]) else None,
            "Banda Inf 2": round(row["Banda Inf 2"], 2) if not pd.isna(row["Banda Inf 2"]) else None,
            "Banda Inf 1": round(row["Banda Inf 1"], 2) if not pd.isna(row["Banda Inf 1"]) else None,
            "Banda Sup 1": round(row["Banda Sup 1"], 2) if not pd.isna(row["Banda Sup 1"]) else None,
            "Banda Sup 2": round(row["Banda Sup 2"], 2) if not pd.isna(row["Banda Sup 2"]) else None,
            "Banda Sup 3": round(row["Banda Sup 3"], 2) if not pd.isna(row["Banda Sup 3"]) else None,
        })
    return registros

def _preencher_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche nulos gerados por lag nas rolling windows."""
    colunas_bandas = [
        "Média Móvel", "Desvio Padrão", "Banda Inf 3", "Banda Inf 2", 
        "Banda Inf 1", "Banda Sup 1", "Banda Sup 2", "Banda Sup 3"
    ]
    for col in colunas_bandas:
        if col in df.columns:
            # Preenche backward os valores nulos iniciais
            df[col] = df[col].bfill()
            
    return df
