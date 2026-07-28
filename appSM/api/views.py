import logging
import json
import pandas as pd

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from django.http import JsonResponse

from rest_framework import status
from rest_framework.exceptions import ParseError
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from appSM.infrastructure.db_fetcher import (
    ExternalDataFetcher,
    ExternalDataNotFoundError,
    ExternalDeviceNotFoundError,
)
from appSM.api.serializers import V2ClassificationHistorySerializer, V2DailySerializer, V2MonthlySerializer, V2ClassificationRangeSerializer
from appSM.services import (
    ClassificationHistoryService,
    AnaliseEstatisticaService,
    PredicaoService,
    PHClassificationService,
    ClassificationRangeService,
)

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
            return fetcher.fetch_monthly_history(
                unidade_id=validated_data["unidade_id"],
                dispositivo_id=validated_data.get("dispositivo_id")
            )
        else:
            return fetcher.fetch_daily_history(
                sensor_id=validated_data["sensor_id"]
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


class V2ClassificationHistory(_V2BaseView):
    serializer_class = V2ClassificationHistorySerializer

    @swagger_auto_schema(
        operation_summary="[v2] Classificacao historica para relatorios",
        request_body=V2ClassificationHistorySerializer,
        responses={
            200: openapi.Response(
                "Classificacoes calculadas",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "results": openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    "periodo": openapi.Schema(type=openapi.TYPE_STRING),
                                    "consumo": openapi.Schema(type=openapi.TYPE_NUMBER),
                                    "classificacao": openapi.Schema(type=openapi.TYPE_STRING),
                                },
                            ),
                        )
                    },
                ),
            )
        },
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            resultado = ClassificationHistoryService().processar(validated_data)
            return JsonResponse(resultado, status=status.HTTP_200_OK)
        except (ExternalDataNotFoundError, ExternalDeviceNotFoundError) as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ClassificacaoPH(APIView):
    """
    API para classificação de pH da água usando modelos ML personalizados por cliente.
    
    O backend principal envia o client_id junto com o valor de pH.
    A API carrega o modelo específico do cliente e retorna a classificação.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Classificação de pH da água",
        operation_description=(
            "Classifica um valor de pH usando o modelo de Machine Learning específico do cliente.\n\n"
            "Fluxo:\n"
            "1. Recebe client_id e ph_value no body da requisição\n"
            "2. Carrega modelo do cliente do disco local\n"
            "3. Faz predição com o modelo\n"
            "4. Retorna classificação e confiança\n\n"
            "Categorias de exemplo: 'adequado', 'alerta', 'crítico' (depende do modelo do cliente)"
        ),
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['client_id', 'ph_value'],
            properties={
                'client_id': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description='Identificador do cliente (ex: "sisar")',
                    example='sisar'
                ),
                'ph_value': openapi.Schema(
                    type=openapi.TYPE_NUMBER,
                    description='Valor de pH a ser classificado (0-14)',
                    example=7.2,
                    minimum=0,
                    maximum=14
                )
            }
        ),
        responses={
            200: openapi.Response(
                description='Classificação realizada com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'client_id': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='ID do cliente'
                        ),
                        'ph_value': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Valor de pH classificado'
                        ),
                        'classification': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Classe prevista pelo modelo',
                            enum=['adequado', 'alerta', 'crítico']
                        ),
                        'confidence': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Confiança da predição (0-1), se disponível'
                        ),
                        'model_version': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Versão do modelo utilizado'
                        )
                    }
                ),
                examples={
                    'application/json': {
                        'client_id': 'sisar',
                        'ph_value': 7.2,
                        'classification': 'adequado',
                        'confidence': 0.95,
                        'model_version': 'v1.0.0'
                    }
                }
            ),
            400: openapi.Response(
                description='Requisição inválida (JSON malformado ou campos obrigatórios ausentes)',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro'
                        )
                    }
                )
            ),
            404: openapi.Response(
                description='Modelo não encontrado para o cliente',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Mensagem indicando que modelo não foi encontrado'
                        )
                    }
                )
            ),
            422: openapi.Response(
                description='Dados válidos mas com conteúdo inadequado (ex: ph_value fora da faixa)',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro de validação'
                        )
                    }
                )
            ),
            500: openapi.Response(
                description='Erro interno do servidor',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro interno'
                        )
                    }
                )
            )
        }
    )
    
    def post(self, request):
        try:
            # Validar se body não está vazio
            if not request.body:
                logger.warning("Requisição de classificação de pH recebida sem body")
                return JsonResponse(
                    {'error': 'Body da requisição está vazio. Envie client_id e ph_value.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            data = json.loads(request.body)
            
            # Validar campos obrigatórios
            if 'client_id' not in data:
                logger.warning("Requisição sem client_id")
                return JsonResponse(
                    {'error': 'Campo obrigatório ausente: client_id'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if 'ph_value' not in data:
                logger.warning(f"Requisição sem ph_value para cliente '{data.get('client_id')}'")
                return JsonResponse(
                    {'error': 'Campo obrigatório ausente: ph_value'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            client_id = str(data['client_id']).strip()
            
            try:
                ph_value = float(data['ph_value'])
            except (ValueError, TypeError):
                logger.warning(f"Valor de pH inválido: {data.get('ph_value')}")
                return JsonResponse(
                    {'error': f"ph_value deve ser um número, recebido: {data.get('ph_value')}"}, 
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )
            
            # Validar client_id
            if not client_id:
                return JsonResponse(
                    {'error': 'client_id não pode ser vazio'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Instanciar serviço de classificação
            ph_service = PHClassificationService()
            
            # Fazer classificação
            resultado = ph_service.classify(
                client_id=client_id,
                ph_value=ph_value
            )
            
            logger.info(
                f"pH {ph_value} classificado como '{resultado['classification']}' "
                f"para cliente '{client_id}'"
            )
            
            return JsonResponse(resultado, status=status.HTTP_200_OK)
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido recebido na classificação de pH: {str(e)}")
            return JsonResponse(
                {'error': 'JSON mal formatado. Verifique a sintaxe.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except FileNotFoundError as e:
            logger.error(f"Modelo não encontrado: {str(e)}")
            return JsonResponse(
                {
                    'error': 'Modelo não encontrado para este cliente',
                    'detail': str(e)
                }, 
                status=status.HTTP_404_NOT_FOUND
            )
        except ValueError as e:
            logger.error(f"Erro de validação na classificação de pH: {str(e)}")
            return JsonResponse(
                {'error': str(e)}, 
                status=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
        except Exception as e:
            logger.exception(f"Erro interno na classificação de pH: {str(e)}")
            return JsonResponse(
                {'error': 'Erro interno ao processar classificação. Tente novamente.'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class V2ClassificationRange(_V2BaseView):
    serializer_class = V2ClassificationRangeSerializer

    @swagger_auto_schema(
        operation_summary="[v2] Verificacao de alerta para consumo fora da faixa verde (dia anterior)",
        request_body=V2ClassificationRangeSerializer,
        responses={
            200: openapi.Response(
                "Resultado da verificacao",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "outside_green_range": openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        "severity": openapi.Schema(type=openapi.TYPE_STRING),
                        "classification": openapi.Schema(type=openapi.TYPE_INTEGER),
                        "classification_label": openapi.Schema(type=openapi.TYPE_STRING),
                        "reference_period": openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATE),
                    },
                ),
            )
        }
    )
    def post(self, request):
        validated_data, error_response = self._validate_payload(request)
        if error_response is not None: return error_response

        try:
            result = ClassificationRangeService().processar(
                validated_data["unidade_id"],
                validated_data.get("reference_period"),
            )
            return JsonResponse(result, status=status.HTTP_200_OK)
            
        except (ExternalDataNotFoundError, ExternalDeviceNotFoundError) as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
