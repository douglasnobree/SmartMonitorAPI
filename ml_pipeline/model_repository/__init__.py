"""
Gerenciador de repositório de modelos de Machine Learning.

Este módulo é responsável por:
- Carregar modelos do disco local ou Google Drive
- Gerenciar cache de modelos em memória
- Versionamento de modelos
- Gerenciamento de metadados dos modelos
"""

import logging
import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from django.conf import settings

from ml_pipeline.model_cache import model_cache

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repositório para armazenamento e recuperação de modelos ML.
    
    Suporta armazenamento local e na nuvem (Google Drive).
    Cache automático em memória para performance.
    """
    
    def __init__(self):
        """Inicializa o repositório de modelos."""
        self.models_dir = settings.MODELS_DIR
        self.cache = model_cache
        logger.info(f"ModelRepository inicializado - Base: {self.models_dir}")
    
    def load_model(
        self, 
        client_id: str, 
        model_type: str, 
        version: Optional[str] = None
    ) -> Any:
        """
        Carrega modelo do cache, disco local ou Google Drive.
        
        Fluxo:
        1. Verifica cache em memória
        2. Se não, busca no disco local
        3. Se não, busca no Google Drive (se habilitado)
        4. Adiciona ao cache antes de retornar
        
        Args:
            client_id (str): ID do cliente (ex: 'sisar')
            model_type (str): Tipo do modelo (ex: 'ph_classification')
            version (str, optional): Versão específica. Se None, usa a mais recente.
        
        Returns:
            model: Modelo carregado pronto para uso
        
        Raises:
            FileNotFoundError: Se modelo não existir em nenhum lugar
            Exception: Se falhar ao carregar
        """
        try:
            # 1. Verificar cache em memória
            cached_model, cached_version = self.cache.get(client_id, model_type)
            if cached_model is not None:
                logger.info(
                    f"Modelo '{model_type}' v{cached_version} para cliente "
                    f"'{client_id}' carregado do CACHE"
                )
                return cached_model
            
            # 2. Buscar no disco local
            model_path, metadata = self._get_local_model_path(
                client_id, model_type, version
            )
            
            if model_path and model_path.exists():
                logger.info(
                    f"Carregando modelo '{model_type}' para cliente '{client_id}' "
                    f"do DISCO: {model_path}"
                )
                
                # Carregar modelo com joblib
                model = joblib.load(model_path)
                
                # Adicionar ao cache
                model_version = metadata.get('version', version or 'unknown')
                self.cache.set(
                    client_id, 
                    model_type, 
                    model, 
                    model_version,
                    metadata
                )
                
                logger.info(
                    f"Modelo '{model_type}' v{model_version} para cliente "
                    f"'{client_id}' carregado do disco e adicionado ao cache"
                )
                
                return model
            
            # 3. Buscar no Google Drive (se habilitado)
            if settings.GOOGLE_DRIVE_ENABLED:
                logger.info(
                    f"Modelo não encontrado localmente. "
                    f"Tentando buscar no Google Drive..."
                )
                model = self._load_from_drive(client_id, model_type, version)
                
                if model:
                    # Salvar localmente para próximas requisições
                    self._save_local_model(client_id, model_type, model, version or 'latest')
                    
                    # Adicionar ao cache
                    self.cache.set(client_id, model_type, model, version or 'latest')
                    
                    logger.info(
                        f"Modelo '{model_type}' para cliente '{client_id}' "
                        f"baixado do Drive e salvo localmente"
                    )
                    
                    return model
            
            # Modelo não encontrado em nenhum lugar
            raise FileNotFoundError(
                f"Modelo '{model_type}' para cliente '{client_id}' não encontrado. "
                f"Verifique se o modelo está em: {self.models_dir}/{model_type}/client_{client_id}/"
            )
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.exception(
                f"Erro ao carregar modelo '{model_type}' para cliente '{client_id}': {str(e)}"
            )
            raise Exception(f"Falha ao carregar modelo: {str(e)}")
    
    def _get_local_model_path(
        self, 
        client_id: str, 
        model_type: str, 
        version: Optional[str] = None
    ) -> tuple[Optional[Path], Dict[str, Any]]:
        """
        Obtém caminho do modelo no disco local.
        
        Args:
            client_id (str): ID do cliente
            model_type (str): Tipo do modelo
            version (str, optional): Versão específica
        
        Returns:
            tuple: (path_do_modelo, metadados) ou (None, {})
        """
        client_dir = self.models_dir / model_type / f"client_{client_id}"
        
        if not client_dir.exists():
            logger.warning(f"Diretório do cliente não existe: {client_dir}")
            return None, {}
        
        # Se versão não especificada, pegar a mais recente
        if not version:
            model_files = list(client_dir.glob("model_*.joblib"))
            if not model_files:
                logger.warning(f"Nenhum modelo encontrado em: {client_dir}")
                return None, {}
            
            # Ordenar por nome (assume formato model_vX.X.X.joblib)
            model_files.sort(reverse=True)
            model_path = model_files[0]
            
            # Extrair versão do nome do arquivo
            version = model_path.stem.replace('model_', '')
        else:
            model_path = client_dir / f"model_{version}.joblib"
        
        # Carregar metadados se existirem
        metadata_path = client_dir / f"metadata_{version}.json"
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar metadados de {metadata_path}: {e}")
        
        metadata['version'] = version
        
        return model_path, metadata
    
    def _load_from_drive(
        self, 
        client_id: str, 
        model_type: str, 
        version: Optional[str] = None
    ) -> Optional[Any]:
        """
        Carrega modelo do Google Drive.
        
        Args:
            client_id (str): ID do cliente
            model_type (str): Tipo do modelo
            version (str, optional): Versão específica
        
        Returns:
            model: Modelo carregado ou None se não encontrado
        """
        # TODO: Implementar integração com Google Drive na Fase 2
        logger.warning("Google Drive ainda não implementado - retornando None")
        return None
    
    def _save_local_model(
        self, 
        client_id: str, 
        model_type: str, 
        model: Any, 
        version: str
    ) -> Path:
        """
        Salva modelo no disco local.
        
        Args:
            client_id (str): ID do cliente
            model_type (str): Tipo do modelo
            model (Any): Objeto do modelo
            version (str): Versão do modelo
        
        Returns:
            Path: Caminho onde o modelo foi salvo
        """
        client_dir = self.models_dir / model_type / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = client_dir / f"model_{version}.joblib"
        
        # Salvar modelo com joblib
        joblib.dump(model, model_path)
        
        logger.info(f"Modelo salvo em: {model_path}")
        
        return model_path
    
    def invalidate_cache(
        self, 
        client_id: str, 
        model_type: Optional[str] = None
    ) -> None:
        """
        Invalida cache de um cliente específico.
        
        Útil quando modelo é atualizado e precisa ser recarregado.
        
        Args:
            client_id (str): ID do cliente
            model_type (str, optional): Tipo específico do modelo
        """
        self.cache.invalidate(client_id, model_type)
        
        suffix = f" - modelo '{model_type}'" if model_type else ""
        logger.info(f"Cache invalidado para cliente '{client_id}'{suffix}")


# Instância global do repositório
model_repository = ModelRepository()


class DriveModelStorage:
    """
    Implementação específica para Google Drive.
    
    Será usado inicialmente para armazenar modelos na nuvem.
    """
    
    def __init__(self, credentials_path: str):
        """
        Inicializa conexão com Google Drive.
        
        Args:
            credentials_path (str): Caminho para arquivo de credenciais
        """
        self.credentials_path = credentials_path
        logger.info("DriveModelStorage inicializado")
        # TODO: Implementar autenticação com Google Drive API
    
    def upload(self, local_path: Path, remote_folder: str) -> str:
        """Faz upload de arquivo para Drive."""
        # TODO: Implementar upload
        raise NotImplementedError("Funcionalidade em desenvolvimento")
    
    def download(self, file_id: str, local_path: Path) -> Path:
        """Faz download de arquivo do Drive."""
        # TODO: Implementar download
        raise NotImplementedError("Funcionalidade em desenvolvimento")
