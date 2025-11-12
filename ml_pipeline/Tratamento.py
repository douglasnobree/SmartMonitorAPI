"""
Interface abstrata para serviços de tratamento de dados.

Este módulo define a interface base que todos os serviços de processamento
de dados devem implementar no pipeline de Machine Learning.
"""

from abc import ABC, abstractmethod


class Tratamento(ABC):
    """
    Classe abstrata base para serviços de tratamento de dados.
    
    Todos os serviços que processam dados (predição, classificação, análise)
    devem herdar desta classe e implementar o método processarDados.
    """

    @abstractmethod
    def processarDados(self, dados_request):
        """
        Processa os dados recebidos e retorna o resultado.
        
        Args:
            dados_request (dict): Dicionário com dados a serem processados.
                                 Formato esperado: {'DD/MM/YYYY': valor_float}
        
        Returns:
            dict ou float: Resultado do processamento (depende da implementação)
        
        Raises:
            Exception: Se houver erro no processamento dos dados
        """
        pass
