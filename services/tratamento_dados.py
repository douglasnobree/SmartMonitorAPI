import pandas as pd
import numpy as np

class Tratamentodados:
    @staticmethod
    def tratamento(dados_json, filtrar_zeros=True):
        """
        Processa os dados recebidos de um JSON.
        
        Args:
            dados_json: O dicionário de dados com datas e consumos
            filtrar_zeros: Se True, remove valores menores ou iguais a 0.1
                           Se False, mantém todos os valores
        """
        df = pd.DataFrame({'Data': dados_json.keys(), 'Consumo': dados_json.values()})

        for indice in df.index:
            if pd.isna(df.at[indice, 'Consumo']):
                df.at[indice, 'Consumo'] = df['Consumo'].median()

        df["Acumulado"] = [np.nan for i in range(len(df))]

        df["Acumulado"] = df['Consumo'].cumsum()

        # Filtrar valores pequenos apenas se o parâmetro filtrar_zeros for True
        if filtrar_zeros:
            df = df.loc[df['Consumo'] > 0.1]

        return df