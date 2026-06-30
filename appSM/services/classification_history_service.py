import calendar
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from appSM.db_fetcher import (
    ExternalDataFetcher,
    ExternalDataNotFoundError,
    dataframe_para_historico,
)
from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import (
    AnaliseEstatisticaService,
)

logger = logging.getLogger(__name__)

class ClassificationHistoryService:
    """Monta series historicas para relatorio e reutiliza o pipeline estatistico."""

    def __init__(
        self,
        fetcher: Optional[ExternalDataFetcher] = None,
        analysis_service_cls=AnaliseEstatisticaService,
    ):
        self.fetcher = fetcher or ExternalDataFetcher()
        self.analysis_service_cls = analysis_service_cls

    def processar(self, validated_data: dict) -> dict:
        if validated_data["type"] == "daily":
            return {"results": self._processar_daily(validated_data)}
        return {"results": self._processar_monthly(validated_data)}

    def _processar_daily(self, validated_data: dict) -> list[dict]:
        data_inicio = validated_data["data_inicio"]
        data_fim = validated_data["data_fim"]

        frame = self.fetcher.fetch_history_daily_report(
            unidade_id=validated_data["unidade_id"],
            data_inicio=data_inicio,
            data_fim=data_fim,
        )
        target_frame = frame.loc[
            (frame.index >= pd.Timestamp(data_inicio))
            & (frame.index <= pd.Timestamp(data_fim) + pd.Timedelta(days=1))
        ]
        if target_frame.empty:
            raise ExternalDataNotFoundError("Nenhum registro encontrado no periodo solicitado")

        logger.info(
            "Classificacao historica diaria: unidade=%s periodo=%s..%s contexto=%s registros=%s",
            validated_data["unidade_id"],
            data_inicio,
            data_fim,
            len(frame),
            len(target_frame),
        )
        return self._classificar_linhas(frame, target_frame, janela=30, periodo_formatter=self._formatar_data)

    def _processar_monthly(self, validated_data: dict) -> list[dict]:
        dia_fechamento = 1
        dispositivo_id = validated_data.get("dispositivo_id")
        if dispositivo_id:
            dia_fechamento = self.fetcher.fetch_dispositivo_dia_fechamento(dispositivo_id)

        periodos = self._periodos_do_ano(validated_data["ano"], dia_fechamento)
        inicio_solicitado = periodos[0][0]
        fim_solicitado_exclusivo = periodos[-1][1] + timedelta(days=1)
        inicio_busca = inicio_solicitado - pd.DateOffset(months=12)

        frame = self.fetcher.fetch_history_monthly_report(
            unidade_id=validated_data["unidade_id"],
            inicio=inicio_busca,
            fim=fim_solicitado_exclusivo,
            dia_fechamento_fatura=dia_fechamento,
        )

        periodos_por_inicio = {pd.Timestamp(inicio): fim for inicio, fim in periodos}
        target_frame = frame.loc[frame.index.isin(periodos_por_inicio.keys())]
        if target_frame.empty:
            raise ExternalDataNotFoundError("Nenhum registro encontrado no periodo solicitado")

        logger.info(
            "Classificacao historica mensal: unidade=%s ano=%s dispositivo=%s ciclo=%s contexto=%s registros=%s",
            validated_data["unidade_id"],
            validated_data["ano"],
            dispositivo_id,
            dia_fechamento,
            len(frame),
            len(target_frame),
        )
        return self._classificar_linhas(
            frame,
            target_frame,
            janela=12,
            periodo_formatter=lambda data: self._formatar_periodo(data, periodos_por_inicio[pd.Timestamp(data)]),
        )

    def _classificar_linhas(self, frame, target_frame, janela: int, periodo_formatter) -> list[dict]:
        historico_completo = dataframe_para_historico(frame)
        service = self.analysis_service_cls(janela=janela)
        resultados = []

        for data, row in target_frame.sort_index().iterrows():
            data_formatada = self._formatar_data(data)
            historico_ate_linha = {
                chave: valor
                for chave, valor in historico_completo.items()
                if pd.to_datetime(chave, format="%d/%m/%Y") <= pd.Timestamp(data)
            }
            classificacao = service.processarDados(historico_ate_linha)
            resultados.append(
                {
                    "periodo": periodo_formatter(data),
                    "consumo": float(classificacao.get("Consumo", row["Consumo"])),
                    "classificacao": (classificacao["Classificação"]),
                }
            )
            logger.debug("Linha classificada: %s -> %s", data_formatada, resultados[-1]["classificacao"])

        return resultados

    @staticmethod
    def _formatar_data(data) -> str:
        return pd.Timestamp(data).strftime("%d/%m/%Y")

    def _formatar_periodo(self, inicio, fim) -> str:
        return f"{self._formatar_data(inicio)} a {self._formatar_data(fim)}"

    @classmethod
    def _periodos_do_ano(cls, ano: int, dia_fechamento: int) -> list[tuple[date, date]]:
        periodos = []
        for mes in range(1, 13):
            if dia_fechamento <= 1:
                inicio = cls._safe_date(ano, mes, 1)
                proximo_mes = inicio + pd.DateOffset(months=1)
                fim = proximo_mes.date() - timedelta(days=1)
            else:
                fim_base = cls._safe_date(ano, mes, dia_fechamento)
                inicio = (pd.Timestamp(fim_base) - pd.DateOffset(months=1)).date()
                fim = fim_base - timedelta(days=1)
            periodos.append((inicio, fim))
        return periodos

    @staticmethod
    def _safe_date(ano: int, mes: int, dia: int) -> date:
        ultimo_dia = calendar.monthrange(ano, mes)[1]
        return date(ano, mes, min(dia, ultimo_dia))
