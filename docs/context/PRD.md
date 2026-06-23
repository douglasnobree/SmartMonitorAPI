# PRD - SmartMonitor API

## Contexto
API REST para analise de consumo de agua e classificacao de qualidade (pH) usando ML e analise estatistica. Consumida por uma API backend principal que expoe resultados no frontend. [Fonte: README]

## Problema resolvido
Disponibilizar servicos de predicao de consumo, classificacao estatistica e classificacao de pH via endpoints HTTP autenticados. [Fonte: README][Fonte: codigo]

## Objetivos
- Fornecer predicao diaria e mensal de consumo com base em historico enviado no request. [Fonte: README][Fonte: codigo]
- Classificar consumo atual com Bandas de Bollinger. [Fonte: README][Fonte: codigo]
- Disponibilizar dados completos das bandas para visualizacao. [Fonte: README][Fonte: codigo]
- Classificar pH por cliente com modelos ML armazenados em disco, com contrato inicial voltado a teste de faixa. [Fonte: README][Fonte: codigo][Fonte: usuario]

## Nao objetivos
- Persistir dados de consumo recebidos em banco. [Fonte: codigo]
- Manter modelos de predicao treinados entre requisicoes. [Fonte: README][Fonte: codigo]
- Refresh token dedicado existe apenas como rota comentada; nao esta exposto nesta versao. [Fonte: codigo]
- Definir neste momento limites formais de payload ou rate limit. [Fonte: usuario]

## Fluxos funcionais
- Predicao de consumo: cliente envia historico (datas -> consumo) -> API treina modelo por requisicao -> retorna proximo valor previsto. [Fonte: README][Fonte: codigo]
- Analise estatistica: cliente envia historico -> API calcula bandas de Bollinger -> retorna classificacao do ultimo ponto. [Fonte: README][Fonte: codigo]
- Dados das bandas: cliente envia historico -> API calcula bandas -> retorna lista de registros com bandas. [Fonte: README][Fonte: codigo]
- Classificacao de pH: cliente envia client_id e ph_value -> API carrega modelo do cliente do disco -> retorna classe e confianca (se disponivel). [Fonte: README][Fonte: codigo]
- Versao v2: cliente envia sensor_id ou unidade_id/dispositivo_id -> API consulta banco externo read-only -> retorna os mesmos resultados de predicao/analise. [Fonte: codigo]
- O fluxo e o contrato de pH sao candidatos a evolucao futura sem garantia de compatibilidade estrita. [Fonte: usuario]

## Regras de negocio
- Predicao: modelo treinado e descartado a cada request. [Fonte: README][Fonte: codigo]
- Analise estatistica: janela diaria 30, mensal 12 (meses) no calculo das bandas. [Fonte: codigo]
- Classificacao pH: modelos por cliente em disco e versao derivada do nome do arquivo. [Fonte: codigo]
- Autenticacao obrigatoria para endpoints de negocio (exceto /token e docs). [Fonte: codigo]
- Historico e ordenado por data (DD/MM/YYYY); duplicatas sao agregadas pela mediana do consumo e lacunas/valores nulos sao preenchidos pela mediana. [Fonte: codigo]

## Casos de uso
- Backend principal solicita predicao diaria para consumo do dia seguinte. [Fonte: README]
- Backend principal solicita classificacao estatistica do ultimo consumo do periodo. [Fonte: README]
- Backend principal solicita dados completos das bandas para graficos. [Fonte: README]
- Backend principal solicita classificacao de pH para cliente especifico. [Fonte: README]

## Restricoes
- Persistencia principal em SQLite (db.sqlite3). [Fonte: codigo]
- Sem modelos Django de dominio definidos (sem tabelas proprias). [Fonte: codigo]
- Requer JSON com dicionario data->valor para predicao e estatistica. [Fonte: codigo]
- A versao v2 depende de EXTERNAL_MYSQL_URL ou EXTERNAL_DB_URL para consultar o banco externo read-only. [Fonte: codigo]

## Requisitos funcionais
- RF-001: Autenticar requisicoes com JWT em header Authorization sem prefixo. [Fonte: codigo]
- RF-002: Calcular predicao diaria com base nos dados fornecidos. [Fonte: codigo]
- RF-003: Calcular predicao mensal com base nos dados fornecidos. [Fonte: codigo]
- RF-004: Classificar consumo diario via Bandas de Bollinger. [Fonte: codigo]
- RF-005: Classificar consumo mensal via Bandas de Bollinger. [Fonte: codigo]
- RF-006: Retornar dados completos das bandas de Bollinger. [Fonte: codigo]
- RF-007: Classificar pH por client_id usando modelo local. [Fonte: codigo]
- RF-008: Fornecer endpoint para obtencao de token JWT. [Fonte: codigo]

## Requisitos nao funcionais
- RNF-001: Operar em Python 3.11 com Django/DRF. [Fonte: README]
- RNF-002: Expor docs interativas via Swagger/Redoc. [Fonte: README][Fonte: codigo]
- RNF-003: Registrar logs em console e arquivos rotativos. [Fonte: codigo]
- RNF-004: Suportar deploy via Docker e Gunicorn. [Fonte: README][Fonte: codigo]
- RNF-005: Observabilidade atual limitada a logs em console e arquivos rotativos; nao ha endpoint Prometheus exposto. [Fonte: codigo]

## Metricas de sucesso
- Latencia e taxa de erro por endpoint em logs/monitoramento. [Fonte: codigo]
- Disponibilidade dos endpoints de predicao e classificacao. [Inferencia]

## Riscos
- Predicao treinada por request pode gerar variabilidade e custo por chamada. [Fonte: README][Fonte: codigo]
- Dependencia de arquivos de modelo locais para pH (impacto em deploys). [Fonte: codigo]
- Modelo de pH atual e inicial/de teste e tende a ser substituido por fluxo posterior. [Fonte: usuario]
- Integracao v2 depende de banco externo de terceiros e pode sofrer mudancas fora do controle da API. [Fonte: usuario]

## Lacunas identificadas
- [GAP] Sem especificacao formal OpenAPI exportada em arquivo. [Fonte: codigo]
- [GAP] Sem definicao explicita de regras de validacao de formato de data no endpoint (existe serializer nao usado). [Fonte: codigo]
- [GAP] Nao ha refresh token exposto; a rota permanece comentada no codigo. [Fonte: codigo]
- [GAP] A versao v2 depende de banco externo somente leitura e nao aparece no README original. [Fonte: codigo]
- [GAP] Nao ha descricao de limites de tamanho de payload ou taxas. [Fonte: codigo]
