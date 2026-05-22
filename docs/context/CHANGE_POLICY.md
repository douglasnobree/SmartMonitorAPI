# Change Policy

Objetivo:
Garantir que código e contexto permaneçam sincronizados.

## Quando atualizar documentação

### Atualizar PRD
Se houve:
- alteração de regra de negócio;
- novo fluxo funcional;
- mudança em comportamento visível;
- inclusão/remoção de endpoint.

### Atualizar RFC
Se houve:
- mudança técnica relevante;
- alteração de contrato;
- integração nova;
- mudança de fluxo interno.

### Criar ou atualizar ADR
Se houve:
- decisão arquitetural;
- substituição tecnológica;
- mudança estrutural.

## Checklist obrigatório antes do merge

[ ] Código atualizado

[ ] Testes atualizados

[ ] PRD revisado (se aplicável)

[ ] RFC revisado (se aplicável)

[ ] ADR criado/atualizado (se aplicável)

[ ] CONTEXT_REPORT revisado

[ ] AGENT_CONTEXT_INDEX revisado

## Regra

Nenhuma mudança deve deixar documentação inconsistente com o comportamento real do sistema.