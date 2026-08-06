from typing import Protocol, Any
from pathlib import Path
import joblib
from functools import lru_cache

class ModelStore(Protocol):
    """
    Contrato para carregamento e salvamento de artefatos de ML.
    """
    def carregar(self, model_id: str) -> Any: ...
    def salvar(self, model_id: str, modelo: Any) -> None: ...
    def existe(self, model_id: str) -> bool: ...
    def versoes(self, model_id: str) -> list[str]: ...


class LocalJobLibModelStore:
    """
    Implementação padrão: carrega/salva modelos .joblib do disco local.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @lru_cache(maxsize=10)
    def _carregar_do_disco(self, path: Path) -> Any:
        return joblib.load(path)

    def carregar(self, model_id: str) -> Any:
        path = self.base_dir / f"{model_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path.name}")
        return self._carregar_do_disco(path)

    def salvar(self, model_id: str, modelo: Any) -> None:
        path = self.base_dir / f"{model_id}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, path)
        self._carregar_do_disco.cache_clear()

    def existe(self, model_id: str) -> bool:
        path = self.base_dir / f"{model_id}.joblib"
        return path.exists()
        
    def versoes(self, model_id: str) -> list[str]:
        return []
