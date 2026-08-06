import logging
import pandas as pd
from datetime import timedelta

from appSM.infrastructure.consumo_repository import ConsumoRepository
from appSM.infrastructure.dispositivo_repository import DispositivoRepository
from appSM.infrastructure.exceptions import DataNotFoundError, DeviceNotFoundError
from appSM.application.exceptions import ConsumoNaoEncontrado, DispositivoNaoEncontrado
from appSM.domain.ciclo_faturamento import periodos_do_ano, agregar_por_ciclo_mensal

logger = logging.getLogger(__name__)

class HistoricoUseCase:
    """Monta series historicas para relatorio e reutiliza o pipeline estatistico."""

    def __init__(self, consumo_repo=None, dispositivo_repo=None):
        self.consumo_repo = consumo_repo or ConsumoRepository()
        self.dispositivo_repo = dispositivo_repo or DispositivoRepository()

    def processar(self, validated_data: dict) -> dict:
        if validated_data["type"] == "daily":
            return {"results": self._processar_daily(validated_data)}
        return {"results": self._processar_monthly(validated_data)}

    def _processar_daily(self, validated_data: dict) -> list[dict]:
        data_inicio = validated_data["data_inicio"]
        data_fim = validated_data["data_fim"]

        try:
            frame = self.consumo_repo.buscar_historico_relatorio_diario(
                unidade_id=validated_data["unidade_id"],
                data_inicio=data_inicio,
                data_fim=data_fim,
            )
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado") from exc

        target_frame = frame.loc[
            (frame.index >= pd.Timestamp(data_inicio))
            & (frame.index <= pd.Timestamp(data_fim) + pd.Timedelta(days=1))
        ]
        if target_frame.empty:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado")

        logger.info(
            "Classificacao historica diaria: unidade=%s periodo=%s..%s contexto=%s registros=%s",
            validated_data["unidade_id"],
            data_inicio,
            data_fim,
            len(frame),
            len(target_frame),
        )
        return self._classificar_linhas(frame, target_frame, is_daily=True, periodo_formatter=self._formatar_data)

    def _processar_monthly(self, validated_data: dict) -> list[dict]:
        dia_fechamento = 1
        dispositivo_id = validated_data.get("dispositivo_id")
        if dispositivo_id:
            try:
                dia_fechamento = self.dispositivo_repo.buscar_dia_fechamento(dispositivo_id)
            except DeviceNotFoundError as exc:
                # Se falhar dispositivo, fallback para 1 (como o repo faz se não existir) 
                # Wait, DispositivoRepository.buscar_dia_fechamento lança DeviceNotFoundError
                raise DispositivoNaoEncontrado("Dispositivo nao encontrado") from exc

        periodos = periodos_do_ano(validated_data["ano"], dia_fechamento)
        inicio_solicitado = periodos[0][0]
        fim_solicitado_exclusivo = periodos[-1][1] + timedelta(days=1)
        inicio_busca = inicio_solicitado - pd.DateOffset(months=12)

        try:
            frame_bruto = self.consumo_repo.buscar_historico_mensal_bruto(
                unidade_id=validated_data["unidade_id"],
                inicio=inicio_busca,
                fim=fim_solicitado_exclusivo,
            )
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado") from exc

        frame = agregar_por_ciclo_mensal(frame_bruto, dia_fechamento)

        periodos_por_inicio = {pd.Timestamp(inicio): fim for inicio, fim in periodos}
        target_frame = frame.loc[frame.index.isin(periodos_por_inicio.keys())]
        if target_frame.empty:
            raise ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado")

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
            is_daily=False,
            periodo_formatter=lambda data: self._formatar_periodo(data, periodos_por_inicio[pd.Timestamp(data)]),
        )

    def _classificar_linhas(self, frame, target_frame, is_daily: bool, periodo_formatter) -> list[dict]:
        from appSM.application.estatistica_use_case import _executar_pipeline
        resultados = []
        janela = 30 if is_daily else 12
        frequencia = "diaria" if is_daily else "mensal"

        for data, row in target_frame.sort_index().iterrows():
            data_formatada = self._formatar_data(data)
            historico_ate_linha = frame.loc[frame.index <= pd.Timestamp(data)]
            
            classificacao = _executar_pipeline(historico_ate_linha, janela=janela, frequencia=frequencia)
            
            resultados.append(
                {
                    "periodo": periodo_formatter(data),
                    "consumo": float(classificacao.get("Consumo", row["Consumo"])),
                    "classificacao": int(classificacao["Classificação"]) if pd.notna(classificacao["Classificação"]) else None
                }
            )
            logger.debug("Linha classificada: %s -> %s", data_formatada, resultados[-1]["classificacao"])

        return resultados

    @staticmethod
    def _formatar_data(data) -> str:
        return pd.Timestamp(data).strftime("%d/%m/%Y")

    def _formatar_periodo(self, inicio, fim) -> str:
        return f"{self._formatar_data(inicio)} a {self._formatar_data(fim)}"
