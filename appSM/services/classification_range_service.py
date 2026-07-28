from datetime import date, timedelta
from appSM.services.classification_history_service import ClassificationHistoryService
from appSM.infrastructure.db_fetcher import ExternalDataNotFoundError

class ClassificationRangeService:
    ALERT_CLASSIFICATIONS = {-2, 1, 2}

    def processar(self, unidade_id: int) -> bool:
        yesterday = date.today() - timedelta(days=1)
        
        validated_data_history = {
            "type": "daily",
            "unidade_id": unidade_id,
            "data_inicio": yesterday,
            "data_fim": yesterday,
        }
        
        resultado = ClassificationHistoryService().processar(validated_data_history)
        results = resultado.get("results", [])
        
        if not results:
            raise ExternalDataNotFoundError("Nenhum registro encontrado no periodo solicitado")
            
        ultima_classificacao = results[-1].get("classificacao")
        
        return ultima_classificacao in self.ALERT_CLASSIFICATIONS
