from datetime import date, timedelta
from appSM.services.classification_history_service import ClassificationHistoryService
from appSM.infrastructure.db_fetcher import ExternalDataNotFoundError

class ClassificationRangeService:
    CLASSIFICATION_METADATA = {
        -2: {
            "outside_green_range": True,
            "severity": "critical",
            "classification_label": "Consumo Muito Abaixo do Esperado",
        },
        -1: {
            "outside_green_range": False,
            "severity": "green",
            "classification_label": "Uso Eficiente",
        },
        0: {
            "outside_green_range": False,
            "severity": "green",
            "classification_label": "Consumo Moderado",
        },
        1: {
            "outside_green_range": True,
            "severity": "warning",
            "classification_label": "Uso Elevado",
        },
        2: {
            "outside_green_range": True,
            "severity": "critical",
            "classification_label": "Consumo Excessivo",
        },
    }

    def processar(self, unidade_id: int, reference_period=None) -> dict:
        target_date = reference_period or date.today() - timedelta(days=1)
        
        validated_data_history = {
            "type": "daily",
            "unidade_id": unidade_id,
            "data_inicio": target_date,
            "data_fim": target_date,
        }
        
        resultado = ClassificationHistoryService().processar(validated_data_history)
        results = resultado.get("results", [])
        
        if not results:
            raise ExternalDataNotFoundError("Nenhum registro encontrado no periodo solicitado")
            
        classification = results[-1].get("classificacao")

        try:
            normalized_classification = int(classification)
        except (TypeError, ValueError) as exc:
            raise ValueError("Classificacao de consumo invalida") from exc

        metadata = self.CLASSIFICATION_METADATA.get(normalized_classification)
        if metadata is None:
            raise ValueError("Classificacao de consumo fora do intervalo esperado")

        return {
            **metadata,
            "classification": normalized_classification,
            "reference_period": target_date.isoformat(),
        }
