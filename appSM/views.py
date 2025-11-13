import json
import logging
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from django.http import JsonResponse

# Serviço predição
from ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service import PredicaoMensal_service
from ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service import PredicaoDiaria_service

# Configure logger
logger = logging.getLogger(__name__)

# ===================================================================================================================================================================================
class PredicaoMensal(APIView):
    """
    API para realizar predição de consumo mensal com base nos dados históricos.
    Utiliza um modelo de Regressão Linear para prever o próximo valor de consumo.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Predição de consumo mensal",
        operation_description="Realiza uma predição do próximo valor de consumo com base nos dados históricos fornecidos. Utiliza um modelo de regressão linear.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description="Dicionário onde as chaves são datas no formato DD/MM/YYYY e os valores são os consumos correspondentes",
            example={"01/06/2025": 120.5, "02/06/2025": 115.2, "03/06/2025": 130.0},
            additional_properties=openapi.Schema(
                type=openapi.TYPE_NUMBER,
                description="Valor de consumo para a data especificada"
            )
        ),
        responses={
            200: openapi.Response(
                description='Predição realizada com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'Prediction': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Valor previsto para o próximo consumo'
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='Erro ao processar a requisição',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro ocorrido'
                        )
                    }
                )
            ),
            500: openapi.Response(description='Erro interno do servidor')
        }
    )

    def post(self, request):
        try:
            data = json.loads(request.body)

            predicao_service = PredicaoMensal_service()

            resultado = predicao_service.processarDados(data)

            return JsonResponse({'Prediction': resultado}, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# ===========================================================================================================================================================================================
class PredicaoDiaria(APIView):
    """
    API para realizar predição de consumo diário com base nos dados históricos.
    Utiliza um modelo de Regressão Linear para prever o próximo valor de consumo.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Predição de consumo diário",
        operation_description="Realiza uma predição do próximo valor de consumo com base nos dados históricos fornecidos. Utiliza um modelo de regressão linear.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description="Dicionário onde as chaves são datas no formato DD/MM/YYYY e os valores são os consumos correspondentes",
            example={"01/06/2025": 20.5, "02/06/2025": 21.2, "03/06/2025": 22.0},
            additional_properties=openapi.Schema(
                type=openapi.TYPE_NUMBER,
                description="Valor de consumo para a data especificada"
            )
        ),
        responses={
            200: openapi.Response(
                description='Predição realizada com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'Prediction': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Valor previsto para o próximo consumo'
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='Erro ao processar a requisição',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro ocorrido'
                        )
                    }
                )
            ),
            500: openapi.Response(description='Erro interno do servidor')
        }
    )

    def post(self, request):
        try:
            data = json.loads(request.body)

            predicao_service = PredicaoDiaria_service()

            resultado = predicao_service.processarDados(data)

            return JsonResponse({'Prediction': resultado}, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    
# Serviço classificação 
# ===================================================================================================================================================================================================
from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import (
    AnaliseEstatisticaService
)
from ml_pipeline.senseFlow_A.classificacao.dadosBandas_service import dadosBandas_service

class Analise_estatistica_mensal(APIView):
    """
    API para classificação do consumo mensal baseado em análise estatística.
    Utiliza bandas de Bollinger para classificar o consumo em diferentes categorias.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Classificação do consumo do mês",
        operation_description="Analisa os dados de consumo e retorna a classificação do último registro baseada em bandas de Bollinger.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description="Dicionário onde as chaves são datas no formato DD/MM/YYYY e os valores são os consumos correspondentes",
            example={"01/06/2025": 120.5, "02/06/2025": 115.2, "03/06/2025": 130.0},
            additional_properties=openapi.Schema(
                type=openapi.TYPE_NUMBER,
                description="Valor de consumo para a data especificada"
            )
        ),
        responses={
            200: openapi.Response(
                description='Classificação realizada com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'Data': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Data do último registro analisado'
                        ),
                        'Consumo': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Valor do consumo do último registro'
                        ),
                        'classificacao': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Classificação do consumo: -2 => Faixa inferior 2; -1 => Faixa inferior 1; 0 => Faixa ideal; 1 => Faixa superior 1; 2 => Faixa superior 2',
                            enum=["Faixa inferior 2", "Faixa inferior 1", "Faixa ideal", "Faixa superior 1", "Faixa superior 2", "Sem classificação"]
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='Erro ao processar a requisição',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro ocorrido'
                        )
                    }
                )
            ),
            500: openapi.Response(description='Erro interno do servidor')
        }
    )
    
    def post(self, request):
        try:
            data = json.loads(request.body)

            analiseEstatisticaMensalService = AnaliseEstatisticaService(janela=12)
            
            classificacao = analiseEstatisticaMensalService.processarDados(data)

            return JsonResponse({
                'Data': classificacao['Data'],
                'Consumo': classificacao['Consumo'],
                'classificacao': classificacao['Classificação']
            }, status=status.HTTP_200_OK)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
# ===================================================================================================================================================================================================
class Analise_estatistica_diaria(APIView):
    """
    API para classificação do consumo diário baseado em análise estatística.
    Utiliza bandas de Bollinger para classificar o consumo em diferentes categorias.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Classificação do consumo diário",
        operation_description="Analisa os dados de consumo e retorna a classificação do último registro baseada em bandas de Bollinger",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description="Dicionário onde as chaves são datas no formato DD/MM/YYYY e os valores são os consumos correspondentes",
            example={"01/06/2025": 120.5, "02/06/2025": 115.2, "03/06/2025": 130.0},
            additional_properties=openapi.Schema(
                type=openapi.TYPE_NUMBER,
                description="Valor de consumo para a data especificada"
            )
        ),
        responses={
            200: openapi.Response(
                description='Classificação realizada com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'Data': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Data do último registro analisado'
                        ),
                        'Consumo': openapi.Schema(
                            type=openapi.TYPE_NUMBER,
                            description='Valor do consumo do último registro'
                        ),
                        'classificacao': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Classificação do consumo: -2 => Faixa inferior 2; -1 => Faixa inferior 1; 0 => Faixa ideal; 1 => Faixa superior 1; 2 => Faixa superior 2',
                            enum=["Faixa inferior 2", "Faixa inferior 1", "Faixa ideal", "Faixa superior 1", "Faixa superior 2", "Sem classificação"]
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='Erro ao processar a requisição',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro ocorrido'
                        )
                    }
                )
            ),
            500: openapi.Response(description='Erro interno do servidor')
        }
    )
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            analiseEstatisticaDiariaService = AnaliseEstatisticaService(janela=30)
            
            classificacao = analiseEstatisticaDiariaService.processarDados(data)

            logger.info(f"Análise diária realizada - Data: {classificacao['Data']}, Classificação: {classificacao['Classificação']}")

            return JsonResponse({ 
                'Data': classificacao['Data'], 
                'Consumo': classificacao['Consumo'], 
                'classificacao': classificacao['Classificação']
            }, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            logger.error("JSON inválido recebido na análise diária")
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            logger.exception(f"Erro na análise diária: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

# ===================================================================================================================================================================================================
class DadosBandas(APIView):
    """
    API para obter os dados das bandas de Bollinger calculadas a partir dos consumos.
    Retorna um conjunto completo de dados processados incluindo médias móveis, desvios padrão e bandas de Bollinger.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Dados das bandas de Bollinger",
        operation_description="Processa os dados de consumo e retorna um conjunto completo de informações sobre as bandas de Bollinger, incluindo média móvel, desvio padrão e classificações.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description="Dicionário onde as chaves são datas no formato DD/MM/YYYY e os valores são os consumos correspondentes",
            example={"01/06/2025": 120.5, "02/06/2025": 115.2, "03/06/2025": 130.0},
            additional_properties=openapi.Schema(
                type=openapi.TYPE_NUMBER,
                description="Valor de consumo para a data especificada"
            )
        ),
        responses={
            200: openapi.Response(
                description='Dados processados com sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'dados': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            description='Lista de registros contendo os dados processados',
                            items=openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    'Data': openapi.Schema(type=openapi.TYPE_STRING, description='Data do registro'),
                                    'Consumo': openapi.Schema(type=openapi.TYPE_NUMBER, description='Valor do consumo'),
                                    'Média Móvel': openapi.Schema(type=openapi.TYPE_NUMBER, description='Média móvel do consumo'),
                                    'Desvio Padrão': openapi.Schema(type=openapi.TYPE_NUMBER, description='Desvio padrão do consumo'),
                                    'Banda Inf 1': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda inferior 1 (Média - 1*Desvio)'),
                                    'Banda Inf 2': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda inferior 2 (Média - 2*Desvio)'),
                                    'Banda Sup 1': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda superior 1 (Média + 1*Desvio)'),
                                    'Banda Sup 2': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda superior 2 (Média + 2*Desvio)'),
                                    'Classificação': openapi.Schema(type=openapi.TYPE_STRING, description='Classificação do consumo')
                                }
                            )
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='Erro ao processar a requisição',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description='Descrição do erro ocorrido'
                        )
                    }
                )
            ),
            500: openapi.Response(description='Erro interno do servidor')
        }
    )
    
    def post(self, request):
        try:
            data = json.loads(request.body)

            dadosBandasService = dadosBandas_service()
            dados = dadosBandasService.processarDados(data)
            
            logger.info(f"Dados de bandas processados - Total de registros: {len(dados)}")
            
            # Retornar apenas os dados
            return JsonResponse({
                'dados': dados
            }, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            logger.error("JSON inválido recebido em dados de bandas")
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            logger.exception(f"Erro ao processar dados de bandas: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

