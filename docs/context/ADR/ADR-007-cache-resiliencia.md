# ADR 007: Implementação de Cache TTL para Resiliência a Falhas do Banco Externo

## Status
Aceito

## Contexto
O SmartMonitorAPI depende fortemente de um banco de dados legado (MySQL) externo para todas as suas análises de consumo, predição estatística e cálculos de bandas de Bollinger.
O acesso constante a essa base tem gerado dois grandes gargalos:
1. Lentidão em chamadas repetidas ao mesmo recurso (por exemplo, predição e histórico efetuados simultaneamente no aplicativo mobile do cliente).
2. Quedas e indisponibilidades frequentes na rede/banco legado, o que ocasiona falhas em cascata para os clientes consumindo a nossa API.

## Decisão
Foi decidido implementar um mecanismo de cache de curta duração (TTL de 5 minutos) na camada de repositórios (`ConsumoRepository`), utilizando o `django.core.cache`. O cache atuará como a primeira linha de defesa contra latência e intermitência do banco de dados legado.

## Consequências
**Positivas:**
- Redução direta na latência média para requests idênticas/sequenciais.
- Suavização dos picos de carga sobre o banco de dados legado.
- Diminuição da taxa de erro 500 no caso de intermitências ultra-rápidas no banco legado (dentro da janela de 5 minutos).

**Negativas:**
- Consumo de memória para armazenar objetos em cache no servidor/redis da aplicação (mitigado pelo curto TTL).
- Desalinhamento temporário de informações: um dado modificado manualmente no banco legado nos últimos 5 minutos pode não refletir instantaneamente.

## Gap Restante (Trabalho Futuro)
Embora o cache atenue picos de carga e falhas breves, ele não protege a aplicação em caso de downtime prolongado. Para o futuro, deve-se considerar a implementação de um padrão Circuit Breaker, para fail-fast gracefulness e retornar respostas cacheadas com avisos de "stale data" em vez de timeouts longos e erros HTTP 500.
