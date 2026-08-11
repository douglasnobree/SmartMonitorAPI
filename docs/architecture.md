# 🏗️ Arquitetura do Sistema e Detalhes de Implementação

Este documento aprofunda os aspectos de engenharia e decisões arquiteturais do projeto SmartMonitor API.

## 🏛️ Divisão em Camadas (Modular Architecture)

A aplicação segue uma separação rigorosa de responsabilidades dividida em 4 camadas orientadas a domínio. Embora possua semelhanças com a *Clean Architecture*, ela foi adaptada para manter alta performance lidando com grandes volumes de dados matemáticos via Pandas.

### 1. API (Presentation)
Responsável unicamente pelos contratos de entrada e saída.
- Valida o payload da requisição usando DRF Serializers.
- Entrega respostas HTTP (200, 400, 404, etc).
- Não contém regra de negócio ou lógica matemática.

### 2. Application (Use Cases)
Orquestra o fluxo de dados entre as pontas.
- Requisita dados históricos para a infraestrutura.
- Envia os dados brutos para o domínio.
- Aplica regras de negócio específicas da aplicação (ex: decidir qual janela de tempo usar).

### 3. Domain (Core/Lógica)
Onde reside toda a "inteligência" da aplicação.
- Completamente isolada, recebe DataFrames de Pandas, aplica tratamentos, limpezas, medianas, resolve _gaps_ e _outliers_ (em `tratamento.py`).
- Hospeda os algoritmos de regressão, agrupamento de ciclos de faturamento e cálculos de bandas de Bollinger.

### 4. Infrastructure (Data Access)
Responsável pela fronteira externa (I/O).
- Conexões com banco de dados externo (ex: `consumo_repository.py`, `dispositivo_repository.py`) via engine configurável (SQLAlchemy).
- Implementa resiliência através de cache de curta duração (TTL).
- Hospeda os modelos persistidos de Machine Learning para a classificação de pH.
- Realiza consultas *read-only* de forma eficiente e devolve os dados em estruturas prontas para a camada de aplicação.

## ⚙️ Fluxo Interno do Pipeline (Predição e ML)

1. A camada **API** recebe a requisição, formata os dados e chama o *Use Case* (Application) adequado.
2. A camada **Application** aciona a **Infrastructure** (ex: `ConsumoRepository`) para carregar a série histórica correspondente em um `DataFrame`, usufruindo de camadas de cache se aplicável.
3. A camada **Application** passa os dados brutos para o **Domain**, que aplica uma composição funcional para normalização: tratando anomalias de faturamento, furos de telemetria e substituindo valores discrepantes.
4. Ainda no **Domain**, os modelos matemáticos (como Regressão Linear e cálculos estatísticos baseados em Bollinger) rodam em memória, devolvendo os dados prontos sem _round-trips_ iterativos.
5. A camada **Application** finaliza a regra de negócio e retorna a resposta para a API formatar o JSON.

## 📁 Estrutura de Diretórios Detalhada

```text
SmartMonitorAPI/
├── appSM/                           # App Django principal
│   ├── api/                         # 1. API: Apresentação e REST
│   │   ├── views.py                 # Endpoints e Controllers
│   │   └── serializers.py           # Serializadores de validação adicionais
│   ├── application/                 # 2. Application: Casos de Uso
│   │   ├── estatistica_use_case.py  # Orquestração de Bandas de Bollinger
│   │   ├── historico_use_case.py    # Orquestração de Histórico de Faturamento
│   │   ├── predicao_use_case.py     # Orquestração de Predição
│   │   └── range_use_case.py        # Orquestração de Alertas Diários
│   ├── domain/                      # 3. Domain: Lógica Core e Matemática
│   │   ├── ciclo_faturamento.py     # Regras de agrupamento de ciclos
│   │   ├── regressao_linear.py      # Regressão Linear sem efeitos colaterais
│   │   └── tratamento.py            # Higienização de Pandas
│   ├── infrastructure/              # 4. Infrastructure: Adaptadores e I/O
│   │   ├── consumo_repository.py    # Acesso a dados com Cache TTL
│   │   ├── dispositivo_repository.py# Repositório de devices
│   │   └── models/                  # Modelos ML (.joblib)
│   ├── tests/                       # Suíte de testes particionada
│   │   ├── test_api.py              # Testes da interface API
│   │   ├── test_use_cases.py        # Testes das regras de negócio orquestradas
│   │   └── utils.py                 # Funções auxiliares e mocks
├── projectSM/                       # Configurações globais Django
│   ├── settings.py                  # Parâmetros, apps e middleware
│   ├── urls.py                      # Roteamento centralizado
│   └── authentication.py            # Customização de tokens JWT
├── static/                          # Arquivos estáticos recolhidos
├── requirements.txt                 # Dependências de ambiente
├── Dockerfile                       # Construção da imagem do container
└── manage.py                        # CLI de controle Django
```
