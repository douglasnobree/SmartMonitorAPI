# PRD - SmartMonitor API (v2)

## Contexto
API REST para análise de consumo de água e classificação de qualidade (pH) usando Machine Learning e análise estatística. Atua sob princípios de Arquitetura Modular integrada de forma segura a um banco de dados externo e é consumida por uma API backend principal que expõe resultados no frontend. [Fonte: README]

## Problema resolvido
Disponibilizar serviços de predição de consumo, classificação estatística de Bollinger e classificação de qualidade de pH via endpoints HTTP REST performáticos e autenticados de forma escalável (Versão 2). [Fonte: README][Fonte: codigo]

## Objetivos
- Fornecer predição diária e mensal de consumo na V2 efetuando consultas diretamente no banco SQL read-only e processando eficientemente em DataFrames nativos. [Fonte: README][Fonte: codigo]
- Classificar o consumo de água usando Bandas de Bollinger para sensores e unidades em geral. [Fonte: README][Fonte: codigo]
- Disponibilizar séries de dados das bandas e relatórios históricos sem perdas nas agregações em lote. [Fonte: README][Fonte: codigo]
- Classificar pH por cliente com modelos locais treinados (`.joblib`) localizados sob a camada de domínio. [Fonte: README][Fonte: codigo][Fonte: usuario]

## Não objetivos
- Reintroduzir ou suportar endpoints obsoletos e depreciados (V1 descontinuada). [Fonte: codigo]
- Persistir dados transicionais na base local do serviço de ML; a fonte de verdade permanece sendo o banco relacional somente leitura. [Fonte: codigo]
- Definir neste momento limites estritos de payload ou rate limiting corporativo. [Fonte: usuario]

## Fluxos Funcionais
- **Predição de Consumo V2 (`/v2/prediction/*`):** Cliente envia `sensor_id` (diário) ou `unidade_id` (mensal) -> API valida payload no serializer -> `ExternalDataFetcher` puxa histórico com *Lazy Engine* como `pd.DataFrame` -> Serviço aciona composição funcional `normalizar_historico` -> Regressão Linear projeta consumo -> Resposta JSON retornada. [Fonte: README][Fonte: codigo]
- **Análise Estatística e Relatórios V2 (`/v2/statistic/*` e `/v2/classification/history`):** Cliente especifica período e unidade -> API consulta histórico acrescido de janela anterior para aquecimento estatístico -> Banda de Bollinger calcula desvios e categoriza -> JSON devolvido ordenado. [Fonte: README][Fonte: codigo]
- **Classificação de pH (`/classify/ph`):** Cliente envia `client_id` e `ph_value` -> API busca e carrega sob demanda modelo em `appSM/domain/models/` -> Executa predição probabilística -> Resposta JSON com classe e versão do arquivo devolvido. [Fonte: README][Fonte: codigo]

## Regras de Negócio
- **Predição e Bandas:** Janela estatística padrão de 30 pontos para leituras diárias e 12 meses para ciclos anuais/mensais.
- **Normalização por Composição:** A série histórica importada via DataFrame é purgada de valores inválidos de tempo, preenchida com a mediana da série (para buracos/outliers faturados) em `appSM/domain/tratamento.py`.
- **Autenticação Mandatória:** Requer Token JWT válido sem prefixo `Bearer` em chamadas HTTP corporativas (exceções concedidas à rota `/token` e interfaces Swagger/Redoc).

## Casos de Uso
- Backend corporativo envia chamada solicitando previsão diária inteligente via identificador único de sensor ou medidor.
- Sistema de faturamento extrai relatório consolidado da classificação de consumos atípicos para a unidade ao fim de cada período (via `/v2/classification/history`).
- Monitor IoT submete leitura ao classificador local de água e avalia segurança e índice no espectro pH.

## Requisitos Funcionais
- **RF-001:** Autenticar requisições usando validação customizada JWT sobre o header `Authorization`.
- **RF-002:** Calcular predições temporais pontuais para escopo diário (`/v2/prediction/daily`).
- **RF-003:** Calcular predições temporais consolidadas em fechamentos de ciclo mensal (`/v2/prediction/monthly`).
- **RF-004:** Categorizar consumos com desvios e Bandas de Bollinger diários (`/v2/statistic/daily`).
- **RF-005:** Categorizar consumos para períodos mensais (`/v2/statistic/monthly`).
- **RF-006:** Renderizar curvas brutas inteiras e séries numéricas das bandas (`/v2/statistic/data`).
- **RF-007:** Avaliar status físico-químico da água em fluxos customizados de cliente (`/classify/ph`).
- **RF-008:** Disponibilizar emissão segura do par access/refresh de tokens (`/token`).
- **RF-009:** Computar e entregar históricos agregados e séries temporais customizadas em lotes para faturamento (`/v2/classification/history`).
- **RF-010:** Verificar se o consumo do dia anterior saiu da faixa verde de normalidade para fins de alertas no backend principal (`/v2/classification/range`).

## Requisitos Não Funcionais
- **RNF-001:** Desenvolvido em Python 3.11 sob Django e DRF, operando sobre uma divisão arquitetural de 4 camadas orientadas a domínio (*Arquitetura Modular*).
- **RNF-002:** Interface auto-explicativa com contratos OpenAPI atualizados interativamente em `/swagger` e `/redoc`.
- **RNF-003:** Sistema confiável de logging configurado enquadrando a camada `appSM.services` a saídas segregadas e rotativas (`smartmonitor.log`, `errors.log`).
- **RNF-004:** Otimização computacional em memória através do processamento nativo e ininterrupto por `pandas.DataFrame`, evadiendo reconversões inter-camadas perdas por serializações em strings/dicts de Python.
- **RNF-005:** Garantia de regressão via testes integrados na suíte Django (`manage.py test`) em 41+ asserções blindadas com *Characterization Tests*.

## Riscos e Mitigações
- **Latência do Banco de Leitura Externo:** Resolvida por conexões SQL administradas sob demanda (*Module-level Lazy Evaluation*) sem engarrafar processos durante inicializações ou execuções locais.
- **Alteração Silenciosa de Regras Matemáticas:** Mitigada com implacáveis testes de caracterização unitariamente injetando datasets controlados na suíte de homologação antes de submissões em produção.
