---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

# Fluxo de Trabalho (Graphify e Documentação)

Antes de responder perguntas sobre o código ou iniciar qualquer alteração:
1. **Consultar o Graphify:** Utilize a skill do Graphify (visualizar o `GRAPH_REPORT.md` ou usar ferramentas do `graphify-out/`) para entender o contexto, dependências e conexões do código.
2. **Consultar Documentação Complementar:** Caso necessário, cheque os arquivos na pasta `/docs/`.
3. Ler obrigatoriamente:
   - /docs/context/PRD.md
   - /docs/context/RFC.md
   - /docs/context/AGENT_CONTEXT_INDEX.md

Durante a alteração:
- Mapear o impacto das mudanças no sistema.

Após a alteração (Atualização Obrigatória):
1. **Atualizar Graphify:** Sempre que houver qualquer modificação no código-fonte, é OBRIGATÓRIO rodar a atualização do Graphify automaticamente para manter o grafo sincronizado e atualizado.
2. Atualizar os documentos necessários na pasta `/docs/`.

Regras estritas:
- Nunca inferir regras sem evidência.
- Nunca remover ADRs antigos.