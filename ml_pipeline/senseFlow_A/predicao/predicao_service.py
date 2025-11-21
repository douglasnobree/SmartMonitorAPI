"""
Serviço unificado de predição de consumo usando Regressão Linear.

Este módulo implementa a predição de consumo (diário ou mensal) baseada em
dados históricos utilizando modelo de Regressão Linear com dados acumulados.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from ml_pipeline.Tratamento import Tratamento
from ..modelos.regressaoLinear import LinearRegression_Acumulado

# Configure logger
logger = logging.getLogger(__name__)


class PredicaoService(Tratamento):
    """
    Serviço unificado para predição de consumo usando Regressão Linear.
    
    Treina um modelo de Regressão Linear com dados acumulados e prevê
    o próximo valor de consumo baseado no histórico fornecido.
    
    Suporta predição diária e mensal com tratamento de dados configurável.
    
    Uso:
        # Predição diária (com validação de percentil)
        service = PredicaoService(tipo='diaria')
        resultado = service.processarDados(dados)
        
        # Predição mensal (sem validação de percentil)
        service = PredicaoService(tipo='mensal')
        resultado = service.processarDados(dados)
    
    Nota: O modelo é treinado e descartado a cada requisição.
    """
    
    # Constantes para tipos de predição
    TIPO_DIARIA = 'diaria'
    TIPO_MENSAL = 'mensal'
    
    def __init__(self, tipo: Optional[str] = None):
        """
        Inicializa o serviço de predição.
        
        Args:
            tipo (str, optional): Tipo de predição ('diaria' ou 'mensal').
                                 Se None, usa 'diaria' como padrão.
        """
        self.tipo = tipo if tipo is not None else self.TIPO_DIARIA
        logger.info(f"PredicaoService inicializado com tipo={self.tipo}")
    
    def processarDados(self, dados_request: Dict[str, float]) -> float:
        """
        Processa dados históricos e retorna predição do próximo consumo.
        
        Args:
            dados_request (dict): Dicionário com datas e consumos históricos.
                                 Formato: {'DD/MM/YYYY': valor_float}
        
        Returns:
            float: Valor previsto para o próximo consumo
        
        Raises:
            ValueError: Se dados_request estiver vazio ou inválido
            Exception: Se houver erro no processamento ou predição
        """
        try:
            if not dados_request:
                raise ValueError("dados_request não pode estar vazio")
            
            # Criação do DataFrame
            df = pd.DataFrame({
                'Data': list(dados_request.keys()), 
                'Consumo': list(dados_request.values())
            })
            
            logger.debug(f"DataFrame criado com {len(df)} registros para predição {self.tipo}")
            # Tratamento de valores nulos - substituir pela mediana
            for indice in df.index:
                if pd.isna(df.at[indice, 'Consumo']):
                    df.at[indice, 'Consumo'] = df['Consumo'].median()
            
            # Cálculo do percentil para tratamento de outliers
            percentile = df['Consumo'].quantile(0.25)
            
            # Para predição diária, validar se percentil é muito baixo
            if self.tipo == self.TIPO_DIARIA and percentile < 1:
                percentile = df['Consumo'].quantile(0.5)
                logger.debug("Percentil ajustado para 0.5 (predição diária)")
            
            mean_value = df['Consumo'].mean()
            
            # Substituir valores abaixo do percentil pela média
            for indice in df.index:
                if df.at[indice, 'Consumo'] < percentile:
                    df.at[indice, 'Consumo'] = mean_value
            
            # Calcular valores acumulados
            df["Acumulado"] = df['Consumo'].cumsum()
            df.reset_index(inplace=True, drop=True)
            
            # Treinar modelo de Regressão Linear
            model = LinearRegression_Acumulado()
            model.train(df)
            
            # Fazer predição para o próximo período
            previsao = model.prediction(len(dados_request))
            
            # Calcular diferença absoluta (consumo previsto para o próximo período)
            resultado = abs(previsao - df['Acumulado'].iloc[-1])
            
            logger.debug(
                f"Predição {self.tipo} calculada: {resultado:.2f} "
                f"(acumulado atual: {df['Acumulado'].iloc[-1]:.2f}, "
                f"próximo acumulado previsto: {previsao:.2f})"
            )
            
            return resultado
            
        except ValueError as e:
            logger.error(f"Erro de validação na predição {self.tipo}: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Erro em PredicaoService ({self.tipo}): {str(e)}")
            raise Exception(str(e))
