from django.contrib import admin
from django.urls import path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

# Import views explicitly
from appSM.views import (
    PredicaoMensal,
    PredicaoDiaria,
    analise_estatistica_diaria,
    analise_estatistica_mensal,
    DadosBandas,
    ClassificacaoPH,
)


schema_view = get_schema_view(
    openapi.Info(
        title="Smart Monitor API",
        default_version='v1',        description="API do projeto Smart Monitor para análise e predição de consumo de água.\n\n"
                   "Esta API oferece serviços para:\n"
                   "- Classificação do consumo atual em categorias baseadas em bandas de Bollinger\n"
                   "- Obtenção de dados estatísticos completos para análise\n"
                   "- Predição de consumo futuro utilizando modelos de aprendizado de máquina\n\n"
                   "Todas as rotas requerem autenticação JWT, que pode ser obtida através do endpoint /token/",
        terms_of_service="https://www.smartmonitor.ifce.edu.br/",
        contact=openapi.Contact(email="RESOURCIFYLTDA@GMAIL.COM"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Rotas da api
    path('admin', admin.site.urls),

    # Root URL - Redirect to Swagger documentation
    path('', schema_view.with_ui('swagger', cache_timeout=0), name='api-root'),
    
    # Rota classificação de consumo atual
    path('statistic/daily', analise_estatistica_diaria.as_view(), name='classificacao-consumo-diaria'),
    path('statistic/monthly', analise_estatistica_mensal.as_view(), name='classificacao-consumo-mensal'),
    path('statistic/data', DadosBandas.as_view(), name='dados-bandas'),

    # Rota de predição de consumo
    path('prediction/monthly', PredicaoMensal.as_view(), name='predicao-consumo-mensal'),
    path('prediction/daily', PredicaoDiaria.as_view(), name='predicao-consumo-diario'),
    
    # Rota de classificação de pH
    path('classify/ph', ClassificacaoPH.as_view(), name='classificacao-ph'),
    
    # Swagger e autenticação
    path('token', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    #path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Documentação da API
    path('swagger', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

]