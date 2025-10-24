import pandas as pd
import numpy as np

from ml_pipeline.Tratamento import Tratamento
from ..modelos.regressaoLinear import LinearRegression_Acumulado

class PredicaoDiaria_service(Tratamento):

    def processarDados(self, dados_request):

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