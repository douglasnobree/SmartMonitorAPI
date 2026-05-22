# Change Impact Map

## /prediction/daily
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py), [ml_pipeline/senseFlow_A/modelos/regressaoLinear.py](ml_pipeline/senseFlow_A/modelos/regressaoLinear.py)
- Servicos chamados: PredicaoService -> LinearRegressionAcumulado. [Fonte: codigo]
- Regras impactadas: treino por request; ajuste diario; input dict data->consumo. [Fonte: codigo]
- Possiveis efeitos colaterais: custo computacional por request; variacao por ordem dos dados. [Inferencia]
- Testes que deveriam existir: validacao de payload, erro JSON invalido, previsao nao negativa, autenticacao obrigatoria. [Inferencia]

## /prediction/monthly
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py), [ml_pipeline/senseFlow_A/modelos/regressaoLinear.py](ml_pipeline/senseFlow_A/modelos/regressaoLinear.py)
- Servicos chamados: PredicaoService -> LinearRegressionAcumulado. [Fonte: codigo]
- Regras impactadas: treino por request; ajuste mensal via residuos; input dict data->consumo. [Fonte: codigo]
- Possiveis efeitos colaterais: sensibilidade a historico curto; custo por request. [Inferencia]
- Testes que deveriam existir: validacao de payload, erro JSON invalido, predicao positiva, autenticacao obrigatoria. [Inferencia]

## /statistic/daily
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Servicos chamados: AnaliseEstatisticaService (janela=30). [Fonte: codigo]
- Regras impactadas: bandas de Bollinger, tratamento de outliers, classificacao do ultimo ponto. [Fonte: codigo]
- Possiveis efeitos colaterais: classificacao altera consumo se outlier; ordem dos dados impacta bandas. [Inferencia]
- Testes que deveriam existir: classificacao por faixa, tratamento de outliers, payload vazio/JSON invalido, autenticacao obrigatoria. [Inferencia]

## /statistic/monthly
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Servicos chamados: AnaliseEstatisticaService (janela=12). [Fonte: codigo]
- Regras impactadas: bandas de Bollinger, classificacao do ultimo ponto. [Fonte: codigo]
- Possiveis efeitos colaterais: janela curta pode afetar estabilidade. [Inferencia]
- Testes que deveriam existir: classificacao mensal, payload vazio/JSON invalido, autenticacao obrigatoria. [Inferencia]

## /statistic/data
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Servicos chamados: AnaliseEstatisticaService.obterDadosCompletos. [Fonte: codigo]
- Regras impactadas: retorno de ate 30 registros, bandas calculadas, preenchimento de nulos. [Fonte: codigo]
- Possiveis efeitos colaterais: volume de resposta; consumo alterado por outlier. [Inferencia]
- Testes que deveriam existir: tamanho maximo de retorno, campos esperados, payload vazio/JSON invalido. [Inferencia]

## /classify/ph
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [appSM/views.py](appSM/views.py), [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)
- Servicos chamados: PHClassificationService.classify -> joblib.load. [Fonte: codigo]
- Regras impactadas: validacao de client_id, conversao de ph_value, busca de modelo por client_id, versao por nome. [Fonte: codigo]
- Possiveis efeitos colaterais: FileNotFoundError para cliente desconhecido; impacto em deploy sem modelos. [Fonte: codigo]
- Testes que deveriam existir: cliente inexistente, ph_value fora de faixa, modelo sem predict_proba, autenticacao obrigatoria. [Inferencia]

## /token
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py)
- Servicos chamados: SimpleJWT TokenObtainPairView. [Fonte: codigo]
- Regras impactadas: formato de request username/password. [Fonte: codigo]
- Possiveis efeitos colaterais: expor refresh token se habilitado. [Inferencia]
- Testes que deveriam existir: credenciais invalidas, formato de resposta. [Inferencia]

## /swagger, /redoc, /
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py)
- Servicos chamados: drf-yasg schema_view. [Fonte: codigo]
- Regras impactadas: disponibilidade de docs sem auth. [Fonte: codigo]
- Possiveis efeitos colaterais: exposicao de contrato publicamente. [Inferencia]
- Testes que deveriam existir: acesso publico, status 200. [Inferencia]

## /admin
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py)
- Servicos chamados: Django admin. [Fonte: codigo]
- Regras impactadas: acesso via auth Django. [Fonte: codigo]
- Possiveis efeitos colaterais: superficie de administracao exposta. [Inferencia]
- Testes que deveriam existir: acesso restrito. [Inferencia]

## Rotas Prometheus (django_prometheus.urls)
- Arquivos afetados: [projectSM/urls.py](projectSM/urls.py), [projectSM/settings.py](projectSM/settings.py)
- Servicos chamados: django_prometheus. [Fonte: codigo]
- Regras impactadas: exposicao de metrics. [Fonte: codigo]
- Possiveis efeitos colaterais: acesso publico a metricas. [Inferencia]
- Testes que deveriam existir: endpoint de metrics acessivel e protegido se necessario. [Inferencia]
