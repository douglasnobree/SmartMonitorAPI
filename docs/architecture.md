# 🏗️ Arquitetura do Sistema e Detalhes de Implementação

Este documento aprofunda os aspectos de engenharia e decisões arquiteturais do projeto SmartMonitor API.

## 🏛️ Divisão em Camadas (Modular Architecture)

A aplicação segue uma separação rigorosa de responsabilidades dividida em 4 camadas orientadas a domínio. Embora possua semelhanças com a *Clean Architecture*, ela foi adaptada para manter alta performance lidando com grandes volumes de dados matemáticos via Pandas.

### 1. API (Presentation)
Responsável unicamente pelos contratos de entrada e saída.
- Valida o payload da requisição usando DRF Serializers.
- Entrega respostas HTTP (200, 400, 404, etc).
- Não contém regra de negócio ou lógica matemática.

### 2. Services (Use Cases)
Orquestra o fluxo de dados entre as pontas.
- Requisita dados históricos para a infraestrutura.
- Envia os dados brutos para o domínio.
- Aplica regras de negócio específicas da aplicação (ex: decidir qual janela de tempo usar).

### 3. Domain (Core/Lógica)
Onde reside toda a "inteligência" da aplicação.
- Completamente isolada, recebe DataFrames de Pandas, aplica tratamentos, limpezas, medianas, resolve _gaps_ e _outliers_ (em `tratamento.py`).
- Hospeda os algoritmos de regressão e cálculos de bandas de Bollinger.
- Hospeda os modelos persistidos de Machine Learning para a classificação de pH.

### 4. Infrastructure (Data Access)
Responsável pela fronteira externa (I/O).
- Conexões com banco de dados externo via *Module-level Lazy Engine* (`db_fetcher.py`).
- Realiza consultas *read-only* de forma eficiente e devolve os dados em estruturas prontas para a camada de serviço.

## ⚙️ Fluxo Interno do Pipeline (Predição e ML)

1. A camada **API** recebe a requisição, formata os dados e chama o Service adequado.
2. A camada de **Services** aciona a **Infrastructure** (`ExternalDataFetcher`) para carregar a série histórica correspondente em um `DataFrame`.
3. O Service passa os dados brutos para o **Domain**, que aplica uma composição funcional para normalização: tratando anomalias de faturamento, furos de telemetria e substituindo valores discrepantes.
4. Ainda no **Domain**, os modelos matemáticos (como Regressão Linear Acumulada e cálculos estatísticos baseados em Bollinger) rodam em memória, devolvendo os dados prontos sem _round-trips_ iterativos.
5. O Service finaliza a regra de negócio (ex: mapeando status do usuário) e retorna para a API exibir o JSON formatado.

## 📁 Estrutura de Diretórios Detalhada

```text
SmartMonitorAPI/
├── appSM/                           # App Django principal
│   ├── api/                         # 1. API: Apresentação e REST
│   │   ├── views.py                 # Endpoints DRF V2
│   │   └── serializers.py           # Contratos JSON
│   ├── services/                    # 2. Services: Regras e Casos de Uso
│   │   ├── predicao_service.py      # Predição linear
│   │   ├── analise_estatistica_service.py # Bandas estatísticas
│   │   ├── classification_history_service.py # Histórico de faturamento
│   │   ├── classification_range_service.py   # Alertas diários
│   │   └── ph_classification_service.py      # Qualidade de água
│   ├── domain/                      # 3. Domain: Algoritmos e Lógica ML
│   │   ├── tratamento.py            # Limpeza e preenchimento de gaps
│   │   ├── regressao_linear.py      # Matemática de regressão acumulada
│   │   └── models/                  # Modelos ML persistidos (.joblib)
│   ├── infrastructure/              # 4. Infrastructure: Adaptadores Externos
│   │   └── db_fetcher.py            # Banco de dados read-only
│   ├── tests.py                     # Suíte de testes unitários da V2
│   └── test_characterization.py     # Testes de caracterização (regras invioláveis)
├── projectSM/                       # Configurações globais Django
│   ├── settings.py                  # Parâmetros, apps e middleware
│   ├── urls.py                      # Roteamento centralizado
│   └── authentication.py            # Customização de tokens JWT
├── static/                          # Arquivos estáticos recolhidos
├── requirements.txt                 # Dependências de ambiente
├── Dockerfile                       # Construção da imagem do container
└── manage.py                        # CLI de controle Django
```
