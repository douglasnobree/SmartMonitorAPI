import logging

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from django.http import JsonResponse

from rest_framework import status
from rest_framework.exceptions import ParseError
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from appSM.db_fetcher import ExternalDataFetcher, ExternalDataNotFoundError, dataframe_para_historico
from appSM.serializers import V2DailySerializer, V2MonthlySerializer
from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import AnaliseEstatisticaService
from ml_pipeline.senseFlow_A.predicao.predicao_service import PredicaoService

logger = logging.getLogger(__name__)


class _V2BaseView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = None
    is_monthly = False

    def _validate_payload(self, request):
        try:
            payload = request.data
        except ParseError:
            return None, JsonResponse({"error": "JSON mal formatado."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=payload)
        if not serializer.is_valid():
            return None, JsonResponse({"error": "Parâmetros inválidos", "details": serializer.errors}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        return serializer.validated_data, None

    def _fetch_history(self, validated_data):
        fetcher = ExternalDataFetcher()
        if self.is_monthly:
            # Encaminha unidade_id e o dispositivo_id (caso tenha sido enviado)
            return dataframe_para_historico(
                fetcher.fetch_monthly_history(
                    unidade_id=validated_data["unidade_id"],
                    dispositivo_id=validated_data.get("dispositivo_id")
                )
            )
        else:
            return dataframe_para_historico(
                fetcher.fetch_daily_history(
                    sensor_id=validated_data["sensor_id"]
                )
            )


class V2PredicaoDiaria(_V2BaseView):
    serializer_class = V2DailySerializer

    @swagger_auto_schema(
        operation_summary="[v2] Predição diária por sensor",
        request_body=V2DailySerializer,
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            historico = self._fetch_history(validated_data)
            resultado = PredicaoService(tipo="diaria").processarDados(historico)
            return JsonResponse({"Prediction": resultado}, status=status.HTTP_200_OK)
        except ExternalDataNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class V2PredicaoMensal(_V2BaseView):
    serializer_class = V2MonthlySerializer
    is_monthly = True

    @swagger_auto_schema(
        operation_summary="[v2] Predição mensal por unidade",
        request_body=V2MonthlySerializer,
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            historico = self._fetch_history(validated_data)
            resultado = PredicaoService(tipo="mensal").processarDados(historico)
            return JsonResponse({"Prediction": resultado}, status=status.HTTP_200_OK)
        except ExternalDataNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class V2AnaliseEstatisticaDiaria(_V2BaseView):
    serializer_class = V2DailySerializer

    @swagger_auto_schema(
        operation_summary="[v2] Estatística diária por sensor",
        request_body=V2DailySerializer,
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            historico = self._fetch_history(validated_data)
            resultado = AnaliseEstatisticaService(janela=30).processarDados(historico)
            return JsonResponse({"Data": resultado["Data"], "Consumo": resultado["Consumo"], "classificacao": resultado["Classificação"]}, status=status.HTTP_200_OK)
        except ExternalDataNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class V2AnaliseEstatisticaMensal(_V2BaseView):
    serializer_class = V2MonthlySerializer
    is_monthly = True

    @swagger_auto_schema(
        operation_summary="[v2] Estatística mensal por unidade",
        request_body=V2MonthlySerializer,
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            historico = self._fetch_history(validated_data)
            resultado = AnaliseEstatisticaService(janela=12).processarDados(historico)
            return JsonResponse({"Data": resultado["Data"], "Consumo": resultado["Consumo"], "classificacao": resultado["Classificação"]}, status=status.HTTP_200_OK)
        except ExternalDataNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class V2DadosBandas(_V2BaseView):
    serializer_class = V2DailySerializer # Presumindo que bandas seja diária pelo sensor. Se for mensal, crie uma rota separada ou mude o serializer.

    @swagger_auto_schema(
        operation_summary="[v2] Dados completos das bandas diárias",
        request_body=V2DailySerializer,
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            historico = self._fetch_history(validated_data)
            dados = AnaliseEstatisticaService(janela=30).obterDadosCompletos(historico)
            return JsonResponse({"dados": dados}, status=status.HTTP_200_OK)
        except ExternalDataNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)