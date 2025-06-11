from appSM.views import *
from django.contrib import admin
from django.urls import path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)


schema_view = get_schema_view(
    openapi.Info(
        title="Smart Monitor API",
        default_version='v1',
        description="Documentação da API do projeto Smart Monitor",
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Rotas da api
    path('admin/', admin.site.urls),

    # Rota de análise de dados
    path('statistic/', Analise_estatistica.as_view()),

    # Rota para dados das bandas de Bollinger
    path('statistic/data', dados_bandas.as_view()),

    # Rota para imagem do gráfico de bandas de Bollinger
    path('statistic/graph', grafico_bandas.as_view()),

    # Predição individual do sensor
    path('prediction/', Analise_Predicao.as_view()),

    # Predição mensal. 
    path('prediction/monthly', Analise_predicao_mensal.as_view()),
    
    # Swagger
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    #path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

]