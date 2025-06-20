from rest_framework.response import Response
from rest_framework import status
import pandas as pd

## Essa é outra abordagem de análise estatistica

def analise_estatistica(data):
    try:
        # Criação do DataFrame
        df = pd.DataFrame({"Data": data['Data'], "Consumo": data['Consumo']})

        df["Média Móvel"] = (
            df["Consumo"]
            .rolling(window=30, min_periods=1)
            .mean()
        )

        df["Desvio Padrão"] = (
            df["Consumo"]
            .rolling(window=30, min_periods=1)
            .std()
        )


        df["Banda Inf 3"] = df["Média Móvel"] - 4.5 * df["Desvio Padrão"]
        df["Banda Inf 2"] = df["Média Móvel"] - 3.0 * df["Desvio Padrão"]
        df["Banda Inf 1"] = df["Média Móvel"] - 1.5 * df["Desvio Padrão"]
        df["Banda Sup 1"] = df["Média Móvel"] + 1.5 * df["Desvio Padrão"]
        df["Banda Sup 2"] = df["Média Móvel"] + 3.0 * df["Desvio Padrão"]
        df["Banda Sup 3"] = df["Média Móvel"] + 4.5 * df["Desvio Padrão"]

        # classificação
        def classifica(row):
            c = row["Consumo"]
            bs3, bs2, bs1, sd, bi1, bi2, bi3  = (
                row["Banda Sup 3"],
                row["Banda Sup 2"],
                row["Banda Sup 1"],
                row["Desvio Padrão"],
                row["Banda Inf 1"],
                row["Banda Inf 2"],
                row["Banda Inf 3"]
            )
            if pd.isna(sd):            return None
            if c >= bs2:               return 2
            elif bs2 >= c >= bs1:      return 1
            elif bs1 >= c >= bi1:      return 0
            elif bi1 >= c >= bi2:      return -1
            else:                      return -2

        df["Classificação"] = df.apply(classifica, axis=1)
        
        
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)