import json
from modelosAnalise.tratamento_dados import Tratamentodados
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from appSM.serializers import MySerializer

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import render
from django.http import JsonResponse

from JSONs import test_json
# Serviço predição
from modelosAnalise.RandomForest.randomforest import model_trained_day, predict_next_day
from modelosAnalise.LinearRegression.RegressaoLinear import LinearRegressionPrediction, LinearRegression_Mensal

class Analise_Predicao(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            additional_properties=openapi.Schema(type=openapi.TYPE_NUMBER)
        ),
        responses={200: openapi.Response('Success', openapi.Schema(type=openapi.TYPE_OBJECT, properties={'prediction': openapi.Schema(type=openapi.TYPE_NUMBER)}))}
    )

    def post(self, request):
        try:
            # Carregar e validar o JSON
            jsondata = json.loads(request.body)
         
            test_json.model_json(jsondata)
            values=test_json.valores

            if len(values) < 30:
                print(f"Dados insuficientes para prever o consumo do sensor .")
            
            prediction = predict_next_day(model_trained_day, values[-30:])

            return JsonResponse({'Previsão para próximo dia': float(prediction)})
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

class Analise_predicao_mensal(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            additional_properties=openapi.Schema(type=openapi.TYPE_NUMBER)
        ),
        responses={200: openapi.Response('Success', openapi.Schema(type=openapi.TYPE_OBJECT, properties={'prediction': openapi.Schema(type=openapi.TYPE_NUMBER)}))}
    )

    def post(self, request):
        try:
            data = json.loads(request.body)

            tratamento_dados = Tratamentodados()
            dados_dataframe = tratamento_dados.tratamento(data)

            if len(dados_dataframe) < 3 and len(dados_dataframe)>12:
                return JsonResponse({'error': 'A lista deve conter quantidade de dados válidos(lista > 3 e lista < 13)'}, status=400)
            
            modelo = LinearRegression_Mensal()

            # Treinar modelo
            modelo.train(dados_dataframe)

            # Realizar predição
            previsao = modelo.prediction(len(dados_dataframe))

            return JsonResponse({'Predição do consumo do próximo mês': (previsao-dados_dataframe['Acumulado'].iloc[-1])}, status=status.HTTP_200_OK)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    
# Serviço classificação 
from modelosAnalise.StatisticalAnalysis.analiseEstatistica import analise_estatistica

class analise_estatistica_geral(APIView):
    
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            additional_properties=openapi.Schema(type=openapi.TYPE_NUMBER)
        ),
        responses={
            200: openapi.Response(
                'Success',
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={'Classificação geral': openapi.Schema(type=openapi.TYPE_STRING)}
                )
            ),
            400: openapi.Response('Bad Request'),
            500: openapi.Response('Internal Server Error'),
        }
    )

    
    def post(self, request):
        try:
            data = json.loads(request.body)

            tratamento_dados = Tratamentodados()
            dados_dataframe = tratamento_dados.tratamento(data)

            if len(dados_dataframe) != 30:
                return JsonResponse({'error': 'A lista deve conter exatamente 30 dados de consumo.'}, status=400)
            
            classificacao = analise_estatistica(dados_dataframe)

            return JsonResponse({'Classificação geral': classificacao}, status=status.HTTP_200_OK)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

class analise_estatistica_sensor(APIView):
    
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            additional_properties=openapi.Schema(type=openapi.TYPE_NUMBER)
        ),
        responses={200: openapi.Response('Success', openapi.Schema(type=openapi.TYPE_OBJECT, properties={'classificação': openapi.Schema(type=openapi.TYPE_STRING)}))}
    )
    
    def post(self, request):
        try:
            data = json.loads(request.body)

            tratamento_dados = Tratamentodados()
            dados_dataframe = tratamento_dados.tratamento(data)

            if len(dados_dataframe) != 30:
                return JsonResponse({'error': 'A lista deve conter exatamente 30 dados de consumo.'}, status=400)

            classificacao = analise_estatistica(dados_dataframe)

            return JsonResponse({ 'Data': classificacao[-1]['Data'], 'Consumo': classificacao[-1]['Consumo'],'Classificação': classificacao[-1]['Classificação']}, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)