from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pandas as pd
import matplotlib
# Configure Matplotlib para usar um backend não interativo (Agg) adequado para servidores web
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64


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
            bs3, bs2, bs1, sd, bi1, bi2,bi3  = (
                row["Banda Sup 3"],
                row["Banda Sup 2"],
                row["Banda Sup 1"],
                row["Desvio Padrão"],
                row["Banda Inf 1"],
                row["Banda Inf 2"],
                row["Banda Inf 3"]
            )
            if pd.isna(sd):            return "Sem classificação"
            if c >= bs2:               return "Muito Alto"
            elif bs2 >= c >= bs1:      return "Alto"
            elif bs1 >= c >= bi1:      return "Normal"
            elif bi1 >= c >= bi2:      return "Baixo"
            else:                      return "Muito Baixo"

        df["Classificação"] = df.apply(classifica, axis=1)
        
        
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

def gerar_grafico_bollinger(data):
    try:
        # Pegar os mesmos dados que a função analise_estatistica processa
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
            bs3, bs2, bs1, sd, bi1, bi2,bi3  = (
                row["Banda Sup 3"],
                row["Banda Sup 2"],
                row["Banda Sup 1"],
                row["Desvio Padrão"],
                row["Banda Inf 1"],
                row["Banda Inf 2"],
                row["Banda Inf 3"]
            )
            if pd.isna(sd):            return "Sem classificação"
            if c >= bs2:               return "Muito Alto"
            elif bs2 >= c >= bs1:      return "Alto"
            elif bs1 >= c >= bi1:      return "Normal"
            elif bi1 >= c >= bi2:      return "Baixo"
            else:                      return "Muito Baixo"

        df["Classificação"] = df.apply(classifica, axis=1)
        df.dropna(inplace=True)
        
        # Criar o gráfico
        fig, ax = plt.subplots(figsize=(14, 7))

        # 1) linha de consumo
        ax.plot(df['Data'], df['Consumo'], label='Consumo')

        # 2) linha da média móvel
        ax.plot(df['Data'], df['Média Móvel'], label='Média Móvel (30d)', linewidth=1.5)


        # preenchimento de banda => Consumo excessivo
        ax.fill_between(df['Data'],
                        df['Banda Sup 2'],
                        df['Banda Sup 3'],
                        alpha=0.2,
                        label='Muito alto',
                        color='red')

        # preenchimento de banda => Consumo Elevado
        ax.fill_between(df['Data'],
                        df['Banda Sup 2'],
                        df['Banda Sup 1'],
                        alpha=0.2,
                        label='Alto',
                        color='orange'
                        )

        ax.fill_between(df['Data'],
                        df['Banda Sup 1'],
                        df['Banda Inf 1'],
                        alpha=0.2,
                        label='Normal',
                        color='green')

        ax.fill_between(df['Data'],
                        df['Banda Inf 1'],
                        df['Banda Inf 2'],
                        alpha=0.2,
                        label='Baixo',
                        color='orange')

        ax.fill_between(df['Data'],
                        df['Banda Inf 2'],
                        df['Banda Inf 3'],
                        alpha=0.2,
                        label='Muito baixo',
                        color='red')

        ax.scatter(df['Data'],
                df['Consumo'],
                marker='o',
                s=10,
                color='blue',
                )


        # Destaca pontos com economia máxima
        economias_max = df[(df['Classificação'] == 'Muito Baixo') | (df['Classificação'] == 'Baixo')]
        ax.scatter(economias_max['Data'],
                economias_max['Consumo'],
                marker='o',
                s=30,
                color='blue',
                label='Economia')


        # destacar pontos de "Consumo Excessivo"
        excessivos = df[df['Classificação'] == 'Muito Alto']
        ax.scatter(excessivos['Data'],
                excessivos['Consumo'],
                marker='o',
                s=30,
                color='red',
                label='Desperdício')


        # formatação do eixo de datas
        if len(df['Data']) > 30:
            ax.set_xticks(df['Data'][::5])
        else:
            ax.set_xticks(df['Data'])
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.xticks(rotation=45, ha="right")

        # legendas e grid
        ax.set_title('Gráfico de classificação de consumo com bandas de Bollinger')
        ax.set_xlabel('Data')
        ax.set_ylabel('Consumo')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')

        plt.tight_layout()


        # Salvar o gráfico em um buffer de memória
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Codificar a imagem em base64
        imagem_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return imagem_base64
    except Exception as e:
        return str(e)