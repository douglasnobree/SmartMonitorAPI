# Agent Context Index

## Sistema
API REST Django para predição de consumo, análise estatística e classificação de pH, consumida por outro backend, reestruturada sob os princípios de Clean Architecture. [Fonte: README]

## Componentes (Arquitetura em Camadas)
- Camada de Apresentação (API): [projectSM/urls.py](projectSM/urls.py), [appSM/api/views.py](appSM/api/views.py) e [appSM/api/serializers.py](appSM/api/serializers.py)
- Camada de Serviços (Casos de Uso):
  - Predição linear: [appSM/services/predicao_service.py](appSM/services/predicao_service.py)
  - Análise estatística: [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
  - Classificação de pH: [appSM/services/ph_classification_service.py](appSM/services/ph_classification_service.py)
  - Relatório histórico de classificação: [appSM/services/classification_history_service.py](appSM/services/classification_history_service.py)
- Camada de Domínio (Lógica e Modelos de ML):
  - Modelos matemáticos: [appSM/domain/regressao_linear.py](appSM/domain/regressao_linear.py)
  - Módulo de tratamento e normalização funcional: [appSM/domain/tratamento.py](appSM/domain/tratamento.py)
  - Modelos joblib persistidos: `appSM/domain/models/`
- Camada de Infraestrutura:
  - Busca de dados externos com Module-level Lazy Engine: [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py)
- Autenticação: [projectSM/authentication.py](projectSM/authentication.py)
- Configuração central: [projectSM/settings.py](projectSM/settings.py)

## Onde encontrar regras
- Regras de predição e cálculos regressivos: [appSM/services/predicao_service.py](appSM/services/predicao_service.py) e [appSM/domain/regressao_linear.py](appSM/domain/regressao_linear.py)
- Regras de normalização, tratamento de gaps e preenchimento de mediana: [appSM/domain/tratamento.py](appSM/domain/tratamento.py)
- Regras de classificação estatística (Bandas de Bollinger): [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
- Regras de classificação de pH: [appSM/services/ph_classification_service.py](appSM/services/ph_classification_service.py)
- Regras de autenticação: [projectSM/authentication.py](projectSM/authentication.py)

## Onde modificar cada comportamento
- Rotas, permissões e visibilidade do Swagger: [projectSM/urls.py](projectSM/urls.py)
- Validações, serialização de payload e erros HTTP: [appSM/api/views.py](appSM/api/views.py) e [appSM/api/serializers.py](appSM/api/serializers.py)
- Predição diária/mensal v2: [appSM/services/predicao_service.py](appSM/services/predicao_service.py)
- Classificação estatística: [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
- Classificação de pH: [appSM/services/ph_classification_service.py](appSM/services/ph_classification_service.py)
- Adaptadores de banco de dados e conector SQL: [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py)
- Auth JWT: [projectSM/authentication.py](projectSM/authentication.py) e SimpleJWT settings em [projectSM/settings.py](projectSM/settings.py)

## Dependências
- Runtime e libs principais: [requirements.txt](requirements.txt) (pandas, scikit-learn, djangorestframework, drf-yasg, django-cors-headers, psycopg2-binary, SQLAlchemy)
- Alternativas por ambiente: [requirements/base.txt](requirements/base.txt), [requirements/development.txt](requirements/development.txt), [requirements/production.txt](requirements/production.txt)

## Arquivos críticos e Suíte de Testes
- Configuração: [projectSM/settings.py](projectSM/settings.py) e [projectSM/urls.py](projectSM/urls.py)
- API Principal (V2): [appSM/api/views.py](appSM/api/views.py) e [appSM/api/serializers.py](appSM/api/serializers.py)
- Serviços: [appSM/services/predicao_service.py](appSM/services/predicao_service.py) e [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
- Testes unitários e de API: [appSM/tests.py](appSM/tests.py)
- Testes de caracterização (regras ML invioláveis): [appSM/test_characterization.py](appSM/test_characterization.py)

## Fluxos importantes
- Predição / Estatística V2: Request -> Validação em serializers -> Consulta a banco em `db_fetcher` (retorna DataFrame puro) -> Serviço executa `normalizar_historico` -> Modelo realiza inferência -> Resposta JSON.
- Classificação pH: Request -> Carga sob demanda do modelo `.joblib` em `appSM/domain/models` -> Predict -> Resposta JSON.
- Autenticação: JWT no header Authorization sem prefixo `Bearer`.

## Ordem recomendada de leitura
1. [README.md](README.md)
2. [projectSM/urls.py](projectSM/urls.py)
3. [appSM/api/views.py](appSM/api/views.py) e [appSM/api/serializers.py](appSM/api/serializers.py)
4. [appSM/domain/tratamento.py](appSM/domain/tratamento.py)
5. [appSM/services/predicao_service.py](appSM/services/predicao_service.py)
6. [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
7. [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py)
8. [appSM/test_characterization.py](appSM/test_characterization.py)
