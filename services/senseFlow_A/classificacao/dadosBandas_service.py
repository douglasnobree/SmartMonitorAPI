import pandas as pd
from services.Tratamento import Tratamento

class dadosBandas_service(Tratamento):

    def processarDados(self, dados_request):

        try:
            # Criação do DataFrame
            df = pd.DataFrame({'Data': list(dados_request.keys()), 'Consumo': list(dados_request.values())})

            janela = 30

            df["Média Móvel"] = (
                df["Consumo"]
                .rolling(window=janela, min_periods=1)
                .mean()
            )

            df["Desvio Padrão"] = (
                df["Consumo"]
                .rolling(window=janela, min_periods=1)
                .std()
            )

            df["Banda Inf 3"] = df["Média Móvel"] - 3 * df["Desvio Padrão"]
            df["Banda Inf 2"] = df["Média Móvel"] - 2 * df["Desvio Padrão"]
            df["Banda Inf 1"] = df["Média Móvel"] - 1 * df["Desvio Padrão"]
            df["Banda Sup 1"] = df["Média Móvel"] + 1 * df["Desvio Padrão"]
            df["Banda Sup 2"] = df["Média Móvel"] + 2 * df["Desvio Padrão"]
            df["Banda Sup 3"] = df["Média Móvel"] + 3 * df["Desvio Padrão"]

            for col in df.columns:
                if col != 'Data' and col != 'Classificação':
                    df[col] = df[col].fillna(0)
                elif col == 'Classificação':
                    df[col] = df[col].fillna("Sem classificação")

            # Retornar o DataFrame com as bandas
            return df.to_dict(orient='records')

        except Exception as e:
            print("Error in dadosBandas_service:", str(e))
            raise Exception(str(e))