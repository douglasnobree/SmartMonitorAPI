---
description: Describe when these instructions should be loaded by the agent based on task context
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

# Fluxo de Trabalho Obrigatório do Agente (Graphify e Documentação)

## 1. Regra de Consulta Prévia (Graphify em Primeiro Lugar)
Sempre que o usuário:
- **Perguntar algo sobre o código** (arquitetura, funcionamento, regras, dependências, rotas, modelos, etc.):
- **Solicitar qualquer alteração ou implementação no código**:

O agente deve **obrigatoriamente checar o Graphify em primeiro lugar** antes de responder ou iniciar a modificação:
1. **Consultar o Graphify:** Inspecione os dados do grafo (`graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md` ou via `/graphify query` / skill `graphify`) para entender a topologia, comunidades, God Nodes, chamadas entre componentes e acoplamento.
2. **Consultar Documentação Complementar (`docs/`) Apenas se Necessário:** O agente só deverá checar os arquivos em `docs/` (como `PRD.md`, `RFC.md`, `ADRs` e `AGENT_CONTEXT_INDEX.md`) se for estritamente necessário para dirimir dúvidas funcionais específicas ou contratos detalhados não cobertos pelo grafo.

---

## 2. Atualização Automática do Graphify (Pós-Alteração de Código)
Sempre que o agente realizar **qualquer alteração no código-fonte**:
1. **Atualização Automática Imediata:** O agente deve obrigatoriamente executar a atualização do Graphify de forma automática para manter `graph.json`, `GRAPH_REPORT.md` e a visualização `graph.html` sincronizados com as mudanças antes de finalizar a resposta.
2. **Documentação:** Atualizar apenas a documentação em `docs/` que tenha sofrido impacto direto pela alteração (respeitando as políticas de mudança).

---

## 3. Regras Estritas de Governança
- **Nunca** responder suposições sobre o fluxo do código sem antes verificar o grafo do Graphify.
- **Nunca** finalizar uma tarefa de alteração de código sem atualizar o Graphify.
- **Nunca** remover ou sobrescrever ADRs históricos em `docs/context/ADR/`.
- Manter o grafo consistente e atualizado para todas as interações.