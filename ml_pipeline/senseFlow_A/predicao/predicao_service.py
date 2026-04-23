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
            
            # Criação do DataFrame bruto (sem pré-processamento)
            df = pd.DataFrame({
                'Data': list(dados_request.keys()), 
                'Consumo': list(dados_request.values())
            })
            logger.debug(f"DataFrame criado com {len(df)} registros para predição {self.tipo}")
            
            # Delegar treinamento ao modelo (modelo faz seu próprio pré-processamento)
            self.modelo.treinar(df)
            
            # Delegar predição ao modelo
            # O modelo já aplica ajustes baseados no tipo_predicao configurado
            resultado = self.modelo.prever(len(dados_request))
            
            logger.debug(f"Predição {self.tipo} calculada: {resultado:.2f}")
            
            return resultado
            
        except ValueError as e:
            logger.error(f"Erro de validação na predição {self.tipo}: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Erro em PredicaoService ({self.tipo}): {str(e)}")
            raise Exception(str(e))
