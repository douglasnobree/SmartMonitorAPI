# Change Impact Map (Arquitetura V2 & Clean Layers)

## /v2/prediction/daily & /v2/prediction/monthly
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [appSM/api/views.py](appSM/api/views.py), [appSM/api/serializers.py](appSM/api/serializers.py), [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py), [appSM/services/predicao_service.py](appSM/services/predicao_service.py), [appSM/domain/regressao_linear.py](appSM/domain/regressao_linear.py), [appSM/domain/tratamento.py](appSM/domain/tratamento.py)
- **Serviços chamados:** `ExternalDataFetcher` -> `PredicaoService` (composição com `normalizar_historico`) -> `LinearRegressionAcumulado`.
- **Regras impactadas:** Busca nativa de DataFrame no banco externo com *Module-level Lazy Engine*; normalização de datas e preenchimento de gaps por mediana; inferência do consumo acumulado.
- **Possíveis efeitos colaterais:** Sensibilidade a indisponibilidade ou latência do banco externo de histórico.
- **Cobertura de Testes Ativa:** Validação de payload por serializers, exceções para sensor/unidade inexistente, autenticação JWT, tratamento de datas e conversão pura em DataFrame testadas e operacionais na suíte de testes (`appSM/tests.py` e `appSM/test_characterization.py`).

## /v2/statistic/daily, /v2/statistic/monthly & /v2/statistic/data
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [appSM/api/views.py](appSM/api/views.py), [appSM/api/serializers.py](appSM/api/serializers.py), [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py), [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py), [appSM/domain/tratamento.py](appSM/domain/tratamento.py)
- **Serviços chamados:** `ExternalDataFetcher` -> `AnaliseEstatisticaService` (composição com `normalizar_historico`).
- **Regras impactadas:** Cálculo de média móvel e desvio padrão (Bandas de Bollinger), tratamento de outliers (faturados substituidos pela mediana da série), fatiamento pandas direto sem round-trip intermediário.
- **Possíveis efeitos colaterais:** Alterações no limiar da janela estatística (30 dias para diário, 12 para mensal) reclassificam séries à beira das faixas de tolerância.
- **Cobertura de Testes Ativa:** Testes de caracterização congelando o cálculo preciso de médias móveis, limites de bandas superior/inferior, validação DRF de request e respostas mockadas limpas.

## /v2/classification/history
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [appSM/api/views.py](appSM/api/views.py), [appSM/api/serializers.py](appSM/api/serializers.py), [appSM/infrastructure/db_fetcher.py](appSM/infrastructure/db_fetcher.py), [appSM/services/classification_history_service.py](appSM/services/classification_history_service.py), [appSM/services/analise_estatistica_service.py](appSM/services/analise_estatistica_service.py)
- **Serviços chamados:** `ClassificationHistoryService` -> `ExternalDataFetcher` -> `AnaliseEstatisticaService`.
- **Regras impactadas:** Montagem de série temporal em lote mantendo histórico anterior ao período solicitado para contexto estatístico de Bollinger (fatiado dinamicamente com indexação Pandas). Suporta fechamentos de ciclo mensal e diário.
- **Possíveis efeitos colaterais:** O consumo de CPU escala proporcionalmente ao número de registros cobrindo o período da consulta.
- **Cobertura de Testes Ativa:** Teste unitário verificando restrição à janela de resposta com reaproveitamento do histórico anterior em memória sem serialização dict extra.

## /classify/ph
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [appSM/api/views.py](appSM/api/views.py), [appSM/services/ph_classification_service.py](appSM/services/ph_classification_service.py)
- **Serviços chamados:** `PHClassificationService.classify` -> `joblib.load` (em `appSM/domain/models/`).
- **Regras impactadas:** Validação de `client_id`, conversão de `ph_value` para float, busca dinâmica de arquivo `.joblib` em disco local com cache da engine ou leitura local.
- **Possíveis efeitos colaterais:** Retorna HTTP 404 caso o diretório do cliente ou o arquivo de modelo não exista em `appSM/domain/models/ph_classification/`.
- **Cobertura de Testes Ativa:** Testada para clientes válidos, exceções de arquivo inexistente (404) e parâmetros inválidos (422/400).

## /token
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [projectSM/authentication.py](projectSM/authentication.py)
- **Serviços chamados:** SimpleJWT `TokenObtainPairView` com autenticação customizada (`JWTAntigravityAuthentication` sem prefixo Bearer).
- **Regras impactadas:** Validação de credenciais e emissão de par de tokens.

## /swagger, /redoc, /
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py)
- **Serviços chamados:** drf-yasg `schema_view`.
- **Regras impactadas:** Documentação interativa publicamente acessível para introspecção de contratos REST.

## /admin
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py)
- **Serviços chamados:** Django admin (autenticação tradicional).

## Rotas Prometheus (django_prometheus.urls)
- **Arquivos afetados:** [projectSM/urls.py](projectSM/urls.py), [projectSM/settings.py](projectSM/settings.py)
- **Serviços chamados:** django_prometheus.
- **Regras impactadas:** Coleta e exposição de métricas de uso HTTP do Django e chamadas do banco para monitoramento de observabilidade.
