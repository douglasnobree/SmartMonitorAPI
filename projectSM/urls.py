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
        default_version='v1',        description="API do projeto Smart Monitor para análise e predição de consumo de energia.\n\n"
                   "Esta API oferece serviços para:\n"
                   "- Classificação do consumo atual em categorias baseadas em bandas de Bollinger\n"
                   "- Obtenção de dados estatísticos completos para análise\n"
                   "- Predição de consumo futuro utilizando modelos de aprendizado de máquina\n\n"
                   "Todas as rotas requerem autenticação JWT, que pode ser obtida através do endpoint /token/",
        terms_of_service="https://www.smartmonitor.com/terms/",
        contact=openapi.Contact(email="contato@smartmonitor.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Rotas da api
    path('admin/', admin.site.urls),

    # Rota classificação de consumo atual
    path('statistic/', Analise_estatistica.as_view(), name='classificacao-consumo'),    # Rota para dados das bandas de Bollinger
    path('statistic/data', dados_bandas.as_view(), name='dados-bandas'),

    # Rota de predição
    path('prediction/', Analise_Predicao.as_view(), name='predicao-consumo'),
    
    # Swagger e autenticação
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    #path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Documentação da API
    path('', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

]