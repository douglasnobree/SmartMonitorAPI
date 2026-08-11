from appSM.domain.tratamento import normalizar_historico
from appSM.domain.outliers import tratar_outliers_iqr
from appSM.domain.regressao_linear import LinearRegressionAcumulado
from appSM.infrastructure.consumo_repository import ConsumoRepository
from appSM.infrastructure.exceptions import DataNotFoundError
from appSM.application.exceptions import ConsumoNaoEncontrado
import pandas as pd
from typing import Optional

class PredicaoUseCase:

    @staticmethod
    def diario(sensor_id: str) -> float:
        """Caso de uso: predição do próximo consumo diário."""
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc
        return _executar_predicao(df[-31:], tipo="diaria", frequencia="diaria")

    @staticmethod
    def mensal(unidade_id: int, dispositivo_id: Optional[str] = None) -> float:
        """Caso de uso: predição do próximo consumo mensal."""
        try:
            df = ConsumoRepository().buscar_historico_mensal(unidade_id, dispositivo_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc
        return _executar_predicao(df, tipo="mensal", frequencia="mensal")

def _executar_predicao(df: pd.DataFrame, tipo: str, frequencia: str) -> float:
    df = normalizar_historico(df, frequencia=frequencia)
    df, _ = tratar_outliers_iqr(df, multiplicador=3.0, substituicao="mediana")
    
    # Remover último período (pode estar incompleto) para evitar viés negativo no modelo
    df = df.iloc[:-1]
    
    modelo = LinearRegressionAcumulado(tipo_predicao=tipo)
    modelo.treinar(df)
    
    try:
        resultado = modelo.prever(len(df))
    except Exception as e:
        raise Exception(f"Erro na predição: {e}") from e
        
    return resultado
