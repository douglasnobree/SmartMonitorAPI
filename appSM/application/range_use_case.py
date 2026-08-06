import logging
from datetime import timedelta
from django.utils import timezone
from appSM.application.historico_use_case import HistoricoUseCase
from appSM.application.exceptions import ConsumoNaoEncontrado
from appSM.domain.classificador import FAIXAS_METADATA

logger = logging.getLogger(__name__)

class RangeUseCase:
    def __init__(self, historico_use_case=None):
        self.historico_use_case = historico_use_case or HistoricoUseCase()

    def processar(self, unidade_id: int, reference_period=None, execution_id=None) -> dict:
        target_date = reference_period or timezone.localdate() - timedelta(days=1)
        flow_id = str(execution_id) if execution_id else "none"
        logger.info(
            "[RANGE_CLASSIFICATION_API] executionId=%s event=analysis_started unitId=%s referencePeriod=%s",
            flow_id,
            unidade_id,
            target_date.isoformat(),
        )
        
        validated_data_history = {
            "type": "daily",
            "unidade_id": unidade_id,
            "data_inicio": target_date,
            "data_fim": target_date,
        }
        
        try:
            resultado = self.historico_use_case.processar(validated_data_history)
        except ConsumoNaoEncontrado as exc:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado") from exc
            
        results = resultado.get("results", [])
        
        if not results:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado")
            
        classification = results[-1].get("classificacao")

        try:
            normalized_classification = int(classification)
        except (TypeError, ValueError) as exc:
            raise ValueError("Classificacao de consumo invalida") from exc

        metadata = FAIXAS_METADATA.get(normalized_classification)
        if metadata is None:
            raise ValueError("Classificacao de consumo fora do intervalo esperado")

        result = {
            **metadata,
            "classification": normalized_classification,
            "reference_period": target_date.isoformat(),
            "execution_id": str(execution_id) if execution_id else None,
        }
        logger.info(
            "[RANGE_CLASSIFICATION_API] executionId=%s event=analysis_completed unitId=%s referencePeriod=%s classification=%s severity=%s outsideGreen=%s",
            flow_id,
            unidade_id,
            target_date.isoformat(),
            normalized_classification,
            result["severity"],
            result["outside_green_range"],
        )
        return result
