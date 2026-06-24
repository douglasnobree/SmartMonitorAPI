# Agent Context Index

## Sistema
API REST Django para predicao de consumo, analise estatistica e classificacao de pH, consumida por outro backend. [Fonte: README]

## Componentes
- API endpoints: [projectSM/urls.py](projectSM/urls.py) e [appSM/views.py](appSM/views.py)
- Predicao: [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py) e [ml_pipeline/senseFlow_A/modelos/regressaoLinear.py](ml_pipeline/senseFlow_A/modelos/regressaoLinear.py)
- Analise estatistica: [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Classificacao pH: [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)
- Autenticacao: [projectSM/authentication.py](projectSM/authentication.py)
- Configuracao: [projectSM/settings.py](projectSM/settings.py)
- Normalizacao de historico: [ml_pipeline/Tratamento.py](ml_pipeline/Tratamento.py)
- Relatorio historico de classificacao: [appSM/services/classification_history_service.py](appSM/services/classification_history_service.py)

## Onde encontrar regras
- Regras de predicao e tratamento: [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py) e [ml_pipeline/senseFlow_A/modelos/regressaoLinear.py](ml_pipeline/senseFlow_A/modelos/regressaoLinear.py)
- Regras de classificacao estatistica: [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Regras de classificacao pH: [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)
- Regras de autenticacao: [projectSM/authentication.py](projectSM/authentication.py)

## Onde modificar cada comportamento
- Rotas e visibilidade: [projectSM/urls.py](projectSM/urls.py)
- Validacoes e erros por endpoint: [appSM/views.py](appSM/views.py)
- Predicao diaria/mensal: [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py)
- Classificacao estatistica: [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- Classificacao de pH: [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)
- Auth JWT: [projectSM/authentication.py](projectSM/authentication.py) e SimpleJWT settings em [projectSM/settings.py](projectSM/settings.py)

## Dependencias
- Runtime e libs principais: [requirements.txt](requirements.txt)
- Alternativas por ambiente: [requirements/base.txt](requirements/base.txt), [requirements/development.txt](requirements/development.txt), [requirements/production.txt](requirements/production.txt)

## Arquivos criticos
- [projectSM/settings.py](projectSM/settings.py)
- [projectSM/urls.py](projectSM/urls.py)
- [appSM/views.py](appSM/views.py)
- [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py)
- [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
- [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)

## Fluxos importantes
- Predicao: request -> parse JSON -> PredicaoService -> resposta. [Fonte: codigo]
- Analise estatistica: request -> analise estatistica -> classificacao -> resposta. [Fonte: codigo]
- Classificacao pH: request -> carga de modelo local -> predict -> resposta. [Fonte: codigo]
- Autenticacao: JWT no header Authorization sem prefixo. [Fonte: codigo]

## Ordem recomendada de leitura
1. [README.md](README.md)
2. [projectSM/urls.py](projectSM/urls.py)
3. [appSM/views.py](appSM/views.py)
4. [ml_pipeline/senseFlow_A/predicao/predicao_service.py](ml_pipeline/senseFlow_A/predicao/predicao_service.py)
5. [ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py](ml_pipeline/senseFlow_A/classificacao/analise_estatistica_service.py)
6. [ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py](ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py)
7. [projectSM/settings.py](projectSM/settings.py)
