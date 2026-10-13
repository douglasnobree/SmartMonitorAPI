# Diretrizes para Agentes de IA - SmartMonitorAPI

## 1. Consulta Obrigatória ao Graphify (Graphify-First)
Sempre que o usuário:
- **Fizer perguntas sobre o código** (arquitetura, conexões, componentes, endpoints, algoritmos, etc.);
- **Solicitar modificações, refatorações ou novas implementações no código**:

O agente deve **obrigatoriamente checar o Graphify em primeiro lugar**:
1. Consultar a topologia e métricas do grafo gerado em `graphify-out/` (`graph.json`, `GRAPH_REPORT.md` ou através da skill `/graphify query` / inspect) para identificar God Nodes, comunidades de componentes, fluxo de chamadas e dependências.
2. **Documentação complementar (`docs/`):** Deve ser consultada **apenas se for estritamente necessário** (ex.: regras de negócio detalhadas no `PRD.md`, contratos de API no `RFC.md` ou decisões de arquitetura em `docs/context/ADR/`).

---

## 2. Atualização Automática Obrigatória do Graphify
Sempre que qualquer alteração de código for realizada (criação, edição ou remoção de arquivos de código ou configuração):
1. **O agente deve atualizar o Graphify de forma automática** antes de concluir seu turno, garantindo que `graph.json`, `GRAPH_REPORT.md` e `graph.html` fiquem perfeitamente sincronizados com o novo estado do código.
2. Atualizar documentos afetados em `docs/` somente se houver impacto direto.

---

## 3. Governança e Regras do Projeto
- Não fazer suposições sobre dependências sem antes validar no grafo do Graphify.
- Nunca remover ou alterar histórico de ADRs existentes em `docs/context/ADR/`.
