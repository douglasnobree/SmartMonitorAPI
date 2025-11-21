"""
Gerenciador de cache de modelos ML em memória.

Este módulo implementa um sistema de cache para modelos de Machine Learning,
evitando o carregamento repetido do disco a cada requisição.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any, Dict

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """
    Gerenciador de cache de modelos em memória com TTL configurável.
    
    Mantém modelos carregados em memória por tempo determinado para
    melhorar performance das requisições de classificação/predição.
    
    Estrutura do cache:
    {
        "client_id": {
            "model_type": {
                "model": <objeto_modelo>,
                "version": "v1.0.0",
                "loaded_at": datetime(...),
                "metadata": {...}
            }
        }
    }
    """
    
    def __init__(self, ttl_minutes: int = 60):
        """
        Inicializa o gerenciador de cache.
        
        Args:
            ttl_minutes (int): Tempo de vida do cache em minutos (padrão: 60)
        """
        self.cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        logger.info(f"ModelCacheManager inicializado com TTL={ttl_minutes} minutos")
    
    def get(
        self, 
        client_id: str, 
        model_type: str
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Obtém modelo do cache se válido.
        
        Args:
            client_id (str): ID do cliente (ex: 'sisar')
            model_type (str): Tipo do modelo (ex: 'ph_classification')
        
        Returns:
            tuple: (modelo, versão) se encontrado e válido, ou (None, None)
        """
        if client_id not in self.cache:
            logger.debug(f"Cliente '{client_id}' não encontrado no cache")
            return None, None
        
        if model_type not in self.cache[client_id]:
            logger.debug(
                f"Modelo '{model_type}' não encontrado no cache "
                f"para cliente '{client_id}'"
            )
            return None, None
        
        cached = self.cache[client_id][model_type]
        
        # Verificar se cache ainda é válido (TTL)
        age = datetime.now() - cached['loaded_at']
        if age < self.ttl:
            logger.debug(
                f"Modelo '{model_type}' para cliente '{client_id}' "
                f"encontrado em cache (idade: {age.total_seconds():.1f}s)"
            )
            return cached['model'], cached['version']
        else:
            # Cache expirado, remover
            logger.info(
                f"Cache expirado para modelo '{model_type}' "
                f"do cliente '{client_id}' (idade: {age})"
            )
            del self.cache[client_id][model_type]
            
            # Remover cliente se não tiver mais modelos em cache
            if not self.cache[client_id]:
                del self.cache[client_id]
            
            return None, None
    
    def set(
        self, 
        client_id: str, 
        model_type: str, 
        model: Any, 
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adiciona modelo ao cache.
        
        Args:
            client_id (str): ID do cliente
            model_type (str): Tipo do modelo
            model (Any): Objeto do modelo carregado
            version (str): Versão do modelo (ex: 'v1.0.0')
            metadata (dict, optional): Metadados adicionais
        """
        if client_id not in self.cache:
            self.cache[client_id] = {}
        
        self.cache[client_id][model_type] = {
            'model': model,
            'version': version,
            'loaded_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        logger.info(
            f"Modelo '{model_type}' v{version} para cliente '{client_id}' "
            f"adicionado ao cache"
        )
    
    def invalidate(
        self, 
        client_id: str, 
        model_type: Optional[str] = None
    ) -> None:
        """
        Remove modelo(s) do cache.
        
        Args:
            client_id (str): ID do cliente
            model_type (str, optional): Tipo específico do modelo.
                                       Se None, remove todos os modelos do cliente.
        """
        if client_id not in self.cache:
            logger.debug(f"Cliente '{client_id}' não encontrado no cache para invalidar")
            return
        
        if model_type:
            # Invalidar modelo específico
            if model_type in self.cache[client_id]:
                del self.cache[client_id][model_type]
                logger.info(
                    f"Cache invalidado para modelo '{model_type}' "
                    f"do cliente '{client_id}'"
                )
                
                # Remover cliente se não tiver mais modelos
                if not self.cache[client_id]:
                    del self.cache[client_id]
        else:
            # Invalidar todos os modelos do cliente
            del self.cache[client_id]
            logger.info(f"Cache invalidado para todos os modelos do cliente '{client_id}'")
    
    def clear(self) -> None:
        """Remove todos os modelos do cache."""
        total_clients = len(self.cache)
        total_models = sum(len(models) for models in self.cache.values())
        
        self.cache.clear()
        
        logger.info(
            f"Cache completamente limpo ({total_clients} clientes, "
            f"{total_models} modelos removidos)"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do cache.
        
        Returns:
            dict: Estatísticas incluindo total de clientes, modelos, etc.
        """
        stats = {
            'total_clients': len(self.cache),
            'total_models': sum(len(models) for models in self.cache.values()),
            'clients': {}
        }
        
        for client_id, models in self.cache.items():
            stats['clients'][client_id] = {
                'total_models': len(models),
                'models': {}
            }
            
            for model_type, cached in models.items():
                age = datetime.now() - cached['loaded_at']
                stats['clients'][client_id]['models'][model_type] = {
                    'version': cached['version'],
                    'age_seconds': age.total_seconds(),
                    'loaded_at': cached['loaded_at'].isoformat()
                }
        
        return stats


# Instância global do cache
model_cache = ModelCacheManager(ttl_minutes=60)
