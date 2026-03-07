"""
Serviço de classificação de pH usando modelos ML personalizados por cliente.

Este módulo implementa a classificação de pH baseada em modelos
treinados específicos para cada cliente.
"""

import logging
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class PHClassificationService:
    """
    Serviço para classificação de pH da água.
    
    Utiliza modelos ML específicos por cliente para classificar
    valores de pH em categorias (ex: adequado, alerta, crítico).
    
    Fluxo:
    1. Recebe client_id e ph_value
    2. Carrega modelo do cliente do disco local
    3. Faz predição com o modelo
    4. Retorna classificação
    """
    
    MODEL_TYPE = 'ph_classification'
    
    def __init__(self):
        """Inicializa o serviço de classificação de pH."""
        self.models_dir = settings.MODELS_DIR
        logger.info("PHClassificationService inicializado")
    
    def _get_model_path(self, client_id: str) -> tuple[Path, Optional[Dict[str, Any]]]:
        """
        Obtém caminho do modelo no disco local.
        
        Args:
            client_id (str): ID do cliente
        
        Returns:
            tuple: (path_do_modelo, metadados) ou raise FileNotFoundError
        """
        client_dir = self.models_dir / self.MODEL_TYPE / f"client_{client_id}"
        
        if not client_dir.exists():
            raise FileNotFoundError(
                f"Diretório do modelo não encontrado: {client_dir}"
            )
        
        # Buscar arquivo de modelo (pega o mais recente se múltiplos)
        model_files = list(client_dir.glob("model_*.joblib"))
        if not model_files:
            raise FileNotFoundError(
                f"Nenhum modelo encontrado para cliente '{client_id}' em: {client_dir}"
            )
        
        # Ordenar por nome e pegar o mais recente
        model_files.sort(reverse=True)
        model_path = model_files[0]
        
        # Extrair versão do nome do arquivo (ex: model_v1.0.0.joblib)
        version = model_path.stem.replace('model_', '')
        
        # Carregar metadados se existirem
        metadata_path = client_dir / f"metadata_{version}.json"
        metadata = {'version': version}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata.update(json.load(f))
            except Exception as e:
                logger.warning(f"Erro ao carregar metadados de {metadata_path}: {e}")
        
        return model_path, metadata
    
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
            
            # Carregar modelo do disco
            model_path, metadata = self._get_model_path(client_id)
            logger.info(f"Carregando modelo de: {model_path}")
            
            model = joblib.load(model_path)
            version = metadata.get('version', 'unknown')
            
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
            
            resultado = {
                'client_id': client_id,
                'ph_value': float(ph_value),
                'classification': str(classification),
                'model_version': version
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
            model_path, metadata = self._get_model_path(client_id)
            model = joblib.load(model_path)
            
            info = {
                'client_id': client_id,
                'model_type': self.MODEL_TYPE,
                'version': metadata.get('version', 'unknown'),
                'model_class': type(model).__name__,
                'model_path': str(model_path),
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
