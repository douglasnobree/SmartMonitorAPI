from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, permissions

from modelosML.StatisticalAnalysis import StatisticalAnalysis


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def helloword(request):
    data = {"Hello, world!"}
    return Response(data, status=status.HTTP_200_OK)


#ANALISE ESTATÍSTICA
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def statis_analy(request):
    response = StatisticalAnalysis(request)
    return response

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def RF():
    return 'RF'