"""
Serviço de predição de consumo diário usando Regressão Linear.

Este módulo implementa a predição de consumo diário baseada em
dados históricos utilizando modelo de Regressão Linear com dados acumulados.
"""

import pandas as pd
import numpy as np

from ml_pipeline.Tratamento import Tratamento
from ..modelos.regressaoLinear import LinearRegression_Acumulado


class PredicaoDiaria_service(Tratamento):
    """
    Serviço para predição de consumo diário.
    
    Treina um modelo de Regressão Linear com dados acumulados e prevê
    o próximo valor de consumo baseado no histórico fornecido.
    
    Nota: O modelo é treinado e descartado a cada requisição.
    """

    def processarDados(self, dados_request):
        """
        Processa dados históricos e retorna predição do próximo consumo.
        
        Args:
            dados_request (dict): Dicionário com datas e consumos históricos.
                                 Formato: {'DD/MM/YYYY': valor_float}
        
        Returns:
            float: Valor previsto para o próximo consumo diário
        
        Raises:
            Exception: Se houver erro no processamento ou predição
        """

        df = pd.DataFrame({'Data': dados_request.keys(), 'Consumo': dados_request.values()})
        

        for indice in df.index:
            if pd.isna(df.at[indice, 'Consumo']):
                df.at[indice, 'Consumo'] = df['Consumo'].median()

        percentile = df['Consumo'].quantile(0.25)
        if percentile < 1:
            percentile = df['Consumo'].quantile(0.5)

        mean_value = df['Consumo'].mean()

        # Substituir valores abaixo do percentil pela média
        for indice in df.index:
            if df.at[indice, 'Consumo'] < percentile:
                df.at[indice, 'Consumo'] = mean_value

        df["Acumulado"] = [np.nan for i in range(len(df))]
        df['Acumulado'] = df['Consumo'].cumsum()
        df.reset_index(inplace=True, drop=True)

        model = LinearRegression_Acumulado()
        model.train(df)
        previsao = model.prediction(len(dados_request))

        resultado = abs(previsao - df['Acumulado'].iloc[-1])

        return resultado