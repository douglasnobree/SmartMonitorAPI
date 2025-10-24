from abc import ABC, abstractmethod

class Tratamento(ABC):

    @abstractmethod
    def processarDados(self, dados_request):
        pass
