"""
Interface abstrata para modelos de predição.

Este módulo define a interface base que todos os modelos de predição
devem implementar, seguindo o Princípio de Inversão de Dependência (DIP).
"""

from abc import ABC, abstractmethod
import pandas as pd


class ModeloPredicao(ABC):
    """
    Interface abstrata para modelos de predição de consumo.
    
    Cada implementação concreta é responsável por:
    1. Fazer seu próprio pré-processamento de dados
    2. Treinar o modelo com os dados preparados
    3. Realizar predições baseadas no modelo treinado
    
    Isso garante baixo acoplamento: serviços dependem da interface,
    não de implementações concretas, permitindo trocar modelos facilmente
    via injeção de dependência.
    """
    
    @abstractmethod
    def treinar(self, dados_brutos: pd.DataFrame) -> None:
        """
        Treina o modelo com dados brutos.
        
        Cada implementação deve fazer seu próprio pré-processamento interno
        (ex: criar colunas adicionais, normalizar, criar features, etc.)
        
        Args:
            dados_brutos (pd.DataFrame): DataFrame com colunas 'Data' e 'Consumo'.
                                         Formato de Data: 'DD/MM/YYYY'
                                         Consumo: valores numéricos (float)
        
        Raises:
            ValueError: Se dados_brutos estiver vazio ou inválido
            Exception: Se houver erro no treinamento do modelo
        """
        pass
    
    @abstractmethod
    def prever(self, n_passos: int) -> float:
        pass
