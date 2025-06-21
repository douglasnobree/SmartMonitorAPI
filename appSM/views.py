import json
from modelosAnalise.tratamento_dados import Tratamentodados
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from django.http import JsonResponse

# Serviço predição
from modelosAnalise.LinearRegression.RegressaoLinear import LinearRegression_Acumulado

class Analise_Predicao(APIView):
    """
    API para realizar predição de consumo com base nos dados históricos.
    Utiliza um modelo de Regressão Linear para prever o próximo valor de consumo.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Predição de consumo",
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

            tratamento_dados = Tratamentodados()
            # Para predição, queremos filtrar valores pequenos
            dados_dataframe = tratamento_dados.tratamento(data, filtrar_zeros=True)
            
            modelo = LinearRegression_Acumulado()

            # Treinar modelo
            modelo.train(dados_dataframe)

            # Realizar predição
            previsao = modelo.prediction(len(dados_dataframe))

            return JsonResponse({'Prediction': (abs(previsao-dados_dataframe['Acumulado'].iloc[-1]))}, status=status.HTTP_200_OK)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    
# Serviço classificação 
from modelosAnalise.StatisticalAnalysis.analiseEstatistica import analise_estatistica

class Analise_estatistica(APIView):
    """
    API para classificação do consumo atual baseado em análise estatística.
    Utiliza bandas de Bollinger para classificar o consumo em diferentes categorias.
    """
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Classificação do consumo atual",
        operation_description="Analisa os dados de consumo e retorna a classificação do último registro baseada em bandas de Bollinger (Economia Máxima, Uso Eficiente, Consumo Moderado, Uso Elevado, Consumo Excessivo).",
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
                            description='Classificação do consumo: -2 => Economia Máxima; -1 => Uso Eficiente, 0 => Consumo Moderado, 1 => Uso Elevado, 2 => Consumo Excessivo)',
                            enum=["Economia Máxima", "Uso Eficiente", "Consumo Moderado", "Uso Elevado", "Consumo Excessivo", "Sem classificação"]
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
        print(self, request)
        try:
            data = json.loads(request.body)

            tratamento_dados = Tratamentodados()
            # Para análise estatística, não queremos filtrar valores pequenos
            dados_dataframe = tratamento_dados.tratamento(data, filtrar_zeros=False)

            classificacao = analise_estatistica(dados_dataframe)

            return JsonResponse({ 'Data': classificacao[-1]['Data'], 'Consumo': classificacao[-1]['Consumo'],'classificacao': classificacao[-1]['Classificação']}, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

class dados_bandas(APIView):
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
                                    'Banda Inf 1': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda inferior 1 (Média - 1.5*Desvio)'),
                                    'Banda Inf 2': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda inferior 2 (Média - 3*Desvio)'),
                                    'Banda Sup 1': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda superior 1 (Média + 1.5*Desvio)'),
                                    'Banda Sup 2': openapi.Schema(type=openapi.TYPE_NUMBER, description='Banda superior 2 (Média + 3*Desvio)'),
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
        print(self, request)
        try:
            data = json.loads(request.body)

            tratamento_dados = Tratamentodados()
            dados_dataframe = tratamento_dados.tratamento(data, filtrar_zeros=False)

            # Obter os dados processados das bandas
            dados = analise_estatistica(dados_dataframe)
            
            # Retornar apenas os dados
            return JsonResponse({
                'dados': dados
            }, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

