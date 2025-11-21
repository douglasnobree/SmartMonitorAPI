"""
Serviço de classificação de pH usando modelos ML personalizados por cliente.

Este módulo implementa a classificação de pH baseada em modelos
treinados específicos para cada cliente.
"""

import logging
import numpy as np
from typing import Dict, Any

from ml_pipeline.model_repository import model_repository

logger = logging.getLogger(__name__)


class PHClassificationService:
    """
    Serviço para classificação de pH da água.
    
    Utiliza modelos ML específicos por cliente para classificar
    valores de pH em categorias (ex: adequado, alerta, crítico).
    
    Fluxo:
    1. Recebe client_id e ph_value
    2. Carrega modelo do cliente (cache → disco → drive)
    3. Faz predição com o modelo
    4. Retorna classificação
    """
    
    MODEL_TYPE = 'ph_classification'
    
    def __init__(self):
        """Inicializa o serviço de classificação de pH."""
        self.repository = model_repository
        logger.info("PHClassificationService inicializado")
    
    def classify(
        self, 
        client_id: str, 
        ph_value: float
    ) -> Dict[str, Any]:
        """
        Classifica valor de pH usando modelo do cliente.
        
        Args:
            client_id (str): ID do cliente (ex: 'sisar')
            ph_value (float): Valor de pH a ser classificado (ex: 7.2)
        
        Returns:
            dict: Resultado da classificação contendo:
                - ph_value (float): Valor de pH fornecido
                - classification (str): Classe prevista
                - confidence (float): Confiança da predição (se disponível)
                - model_version (str): Versão do modelo utilizado
                - client_id (str): ID do cliente
        
        Raises:
            ValueError: Se ph_value for inválido
            FileNotFoundError: Se modelo do cliente não existir
            Exception: Se falhar na classificação
        """
        try:
            # Validar entrada
            if not isinstance(ph_value, (int, float)):
                raise ValueError(
                    f"ph_value deve ser um número, recebido: {type(ph_value).__name__}"
                )
            
            if ph_value < 0 or ph_value > 14:
                logger.warning(
                    f"pH fora da faixa normal (0-14): {ph_value} "
                    f"para cliente '{client_id}'"
                )
            
            logger.info(
                f"Classificando pH {ph_value} para cliente '{client_id}'"
            )
            
            # Carregar modelo do repositório (cache → disco → drive)
            model = self.repository.load_model(
                client_id=client_id,
                model_type=self.MODEL_TYPE
            )
            
            # Preparar input para o modelo
            # Assumindo que o modelo espera array 2D: [[ph_value]]
            X = np.array([[ph_value]])
            
            # Fazer predição
            prediction = model.predict(X)
            classification = prediction[0]
            
            # Tentar obter probabilidades (se o modelo suportar)
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X)
                    confidence = float(np.max(probabilities))
                except Exception as e:
                    logger.debug(f"Modelo não suporta predict_proba: {e}")
            
            # Obter versão do modelo do cache
            _, version = self.repository.cache.get(client_id, self.MODEL_TYPE)
            
            resultado = {
                'client_id': client_id,
                'ph_value': float(ph_value),
                'classification': str(classification),
                'model_version': version or 'unknown'
            }
            
            if confidence is not None:
                resultado['confidence'] = round(confidence, 4)
            
            logger.info(
                f"pH {ph_value} classificado como '{classification}' "
                f"para cliente '{client_id}' "
                f"{f'(confiança: {confidence:.2%})' if confidence else ''}"
            )
            
            return resultado
            
        except ValueError as e:
            logger.error(f"Erro de validação: {str(e)}")
            raise
        except FileNotFoundError as e:
            logger.error(
                f"Modelo não encontrado para cliente '{client_id}': {str(e)}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"Erro ao classificar pH para cliente '{client_id}': {str(e)}"
            )
            raise Exception(f"Falha na classificação: {str(e)}")
    
    def get_model_info(self, client_id: str) -> Dict[str, Any]:
        """
        Obtém informações sobre o modelo do cliente.
        
        Args:
            client_id (str): ID do cliente
        
        Returns:
            dict: Informações do modelo (versão, metadados, etc.)
        """
        try:
            # Carregar modelo para forçar cache
            model = self.repository.load_model(
                client_id=client_id,
                model_type=self.MODEL_TYPE
            )
            
            # Obter informações do cache
            _, version = self.repository.cache.get(client_id, self.MODEL_TYPE)
            
            # Obter metadados se disponíveis
            model_path, metadata = self.repository._get_local_model_path(
                client_id=client_id,
                model_type=self.MODEL_TYPE
            )
            
            info = {
                'client_id': client_id,
                'model_type': self.MODEL_TYPE,
                'version': version or 'unknown',
                'model_class': type(model).__name__,
                'metadata': metadata
            }
            
            # Adicionar informações sobre classes se disponível
            if hasattr(model, 'classes_'):
                info['classes'] = list(model.classes_)
            
            return info
            
        except Exception as e:
            logger.error(
                f"Erro ao obter informações do modelo para cliente '{client_id}': {str(e)}"
            )
            raise
