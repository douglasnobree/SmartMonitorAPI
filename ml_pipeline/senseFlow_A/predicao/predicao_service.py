"""
Serviço unificado de predição de consumo usando Regressão Linear.

Este módulo implementa a predição de consumo (diário ou mensal) baseada em
dados históricos utilizando modelos de Machine Learning com injeção de dependência.

Segue o Princípio de Inversão de Dependência (DIP): o serviço depende da
interface ModeloPredicao, não de implementações concretas, permitindo
trocar modelos facilmente sem modificar o serviço.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from ml_pipeline.Tratamento import Tratamento
from ml_pipeline.modelos.base_modelo import ModeloPredicao
from ..modelos.regressaoLinear import LinearRegressionAcumulado

# Configure logger
logger = logging.getLogger(__name__)


class PredicaoService(Tratamento):
    """
    Serviço unificado para predição de consumo com injeção de dependência.
    
    Este serviço orquestra a predição de consumo usando qualquer modelo
    que implemente a interface ModeloPredicao. O modelo é responsável por
    seu próprio pré-processamento e treinamento.
    
    Suporta predição diária e mensal com ajustes de lógica de negócio.
    
    Uso:
        # Predição com modelo padrão
        service = PredicaoService(tipo='diaria')
        resultado = service.processarDados(dados)
        
        # Predição com modelo customizado (injeção de dependência)
        modelo_custom = MeuModeloPredicao()
        service = PredicaoService(tipo='mensal', modelo=modelo_custom)
        resultado = service.processarDados(dados)
    
    Nota: O modelo é treinado e descartado a cada requisição.
    """
    
    # Constantes para tipos de predição
    TIPO_DIARIA = 'diaria'
    TIPO_MENSAL = 'mensal'
    
    def __init__(self, tipo: Optional[str] = None, modelo: Optional[ModeloPredicao] = None):
        """
        Inicializa o serviço de predição com injeção de dependência.
        
        Args:
            tipo (str, optional): Tipo de predição ('diaria' ou 'mensal').
                                 Se None, usa 'diaria' como padrão.
            modelo (ModeloPredicao, optional): Implementação do modelo de predição.
                                               Se None, usa LinearRegressionAcumulado como padrão.
        """
        self.tipo = tipo if tipo is not None else self.TIPO_DIARIA
        # Se modelo não fornecido, criar com tipo_predicao configurado
        self.modelo = modelo if modelo is not None else LinearRegressionAcumulado(tipo_predicao=self.tipo)
        logger.info(
            f"PredicaoService inicializado com tipo={self.tipo}, "
            f"modelo={type(self.modelo).__name__}"
        )
    
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

            frequencia = self.TIPO_MENSAL if self.tipo == self.TIPO_MENSAL else self.TIPO_DIARIA
            df = self._normalizar_historico(dados_request, frequencia=frequencia)
            logger.debug(
                "Historico normalizado com %s registros para predicao %s",
                len(df),
                self.tipo
            )
            
            df_copiado = df.copy()
            df_tratado, _ = self._tratar_outliers_mediana(df_copiado)
            
            print(f"Dados tratados para predição {self.tipo}:\n{df_tratado}")
            print(f"Dados originais para predição {self.tipo}:\n{df}")
            
            # Delegar treinamento ao modelo (modelo faz seu próprio pré-processamento)
            self.modelo.treinar(df_tratado)
            
            # Delegar predição ao modelo
            # O modelo já aplica ajustes baseados no tipo_predicao configurado
            resultado = self.modelo.prever(len(df))
            
            logger.debug(f"Predição {self.tipo} calculada: {resultado:.2f}")
            
            return resultado
            
        except ValueError as e:
            logger.error(f"Erro de validação na predição {self.tipo}: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Erro em PredicaoService ({self.tipo}): {str(e)}")
            raise Exception(str(e))


    def _tratar_outliers_mediana(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
            """
            Detecta outliers na coluna de consumo e substitui pela mediana.

            A detecção é feita pelo método IQR (3.0 * IQR), com uma faixa mais
            conservadora para evitar que variações normais do histórico sejam
            tratadas como outliers na predição.

            Args:
                df (pd.DataFrame): DataFrame com coluna 'Consumo'

            Returns:
                tuple[pd.DataFrame, pd.Series]: DataFrame com outliers tratados e
                máscara booleana dos pontos tratados
            """
            if df.empty or "Consumo" not in df.columns:
                return df, pd.Series([False] * len(df), index=df.index)

            q1 = df["Consumo"].quantile(0.25)
            q3 = df["Consumo"].quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                return df, pd.Series([False] * len(df), index=df.index)

            multiplicador_iqr = 3.0
            limite_inferior = q1 - multiplicador_iqr * iqr
            limite_superior = q3 + multiplicador_iqr * iqr

            mascara_outliers = (
                (df["Consumo"] < limite_inferior) |
                (df["Consumo"] > limite_superior)
            )

            total_outliers = int(mascara_outliers.sum())
            if total_outliers == 0:
                return df, mascara_outliers

            mediana_referencia = df.loc[~mascara_outliers, "Consumo"].median()
            if pd.isna(mediana_referencia):
                mediana_referencia = df["Consumo"].median()

            df.loc[mascara_outliers, "Consumo"] = mediana_referencia

            logger.info(
                f"Outliers tratados: {total_outliers} valores substituídos "
                f"pela mediana {mediana_referencia:.4f}"
            )

            return df, mascara_outliers