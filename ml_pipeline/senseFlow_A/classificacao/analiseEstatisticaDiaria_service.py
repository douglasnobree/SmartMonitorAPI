"""
Serviço de análise estatística diária usando Bandas de Bollinger.

Este módulo implementa a classificação de consumo diário baseada em
análise estatística utilizando o método de Bandas de Bollinger.
"""

import pandas as pd
import logging

from ml_pipeline.Tratamento import Tratamento

# Configure logger
logger = logging.getLogger(__name__)


class analiseEstatisticaDiaria_service(Tratamento):
    """
    Serviço para classificação de consumo diário usando Bandas de Bollinger.
    
    Analisa os dados de consumo e classifica o último registro em uma de 5 categorias
    baseadas nas bandas de Bollinger calculadas com janela móvel de 30 dias.
    
    Classificações:
        -2: Faixa inferior 2 (muito abaixo do normal)
        -1: Faixa inferior 1 (abaixo do normal)
         0: Faixa ideal (normal)
         1: Faixa superior 1 (acima do normal)
         2: Faixa superior 2 (muito acima do normal)
    """

    def processarDados(self, dados_request):
        """
        Processa dados de consumo diário e retorna a classificação do último registro.
        
        Args:
            dados_request (dict): Dicionário com datas e consumos.
                                 Formato: {'DD/MM/YYYY': valor_float}
        
        Returns:
            dict: Dicionário contendo:
                - Data (str): Data do último registro
                - Consumo (float): Valor do consumo
                - Classificação (int ou str): Classificação do consumo
        
        Raises:
            Exception: Se houver erro no processamento dos dados
        """

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

            # classificação
            def classifica(row):
                """
                Classifica o consumo baseado nas bandas de Bollinger.
                
                Args:
                    row (pd.Series): Linha do DataFrame com valores calculados
                
                Returns:
                    int ou None: Classificação numérica ou None se sem dados suficientes
                """
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
            
            for col in df.columns:
                if col != 'Data' and col != 'Classificação':
                    df[col] = df[col].fillna(0)
                elif col == 'Classificação':
                    df[col] = df[col].fillna("Sem classificação")

            # Get the last row properly using iloc
            last_row = df.iloc[-1]
            classification = {
                "Data": last_row['Data'],
                "Consumo": last_row['Consumo'],
                "Classificação": last_row['Classificação']
            }
            
            logger.debug(f"Classificação calculada para {last_row['Data']}: {classification}")
            return classification
            
        except Exception as e:
            logger.exception(f"Erro em analiseEstatisticaDiaria_service: {str(e)}")
            raise Exception(str(e))
        