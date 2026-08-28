# Workflows locais do runtime WhatsApp

Os exports entram **inativos**. Para importar no n8n local:

```powershell
docker compose --env-file .env.compose --profile workflow-bootstrap run --rm n8n-import
```

`persona-conversation-template.json` é a única fonte canônica para criação e
ressincronização dos workflows `n8n_agents`. O provisionador substitui apenas
persona, agente, webhook e credencial. Prompt, políticas, campos e conhecimento
vêm do Graph JSON publicado e dos `context_cards`; nunca existe função ou
template específico por cliente.

O template `graph_agentic_v3` usa proposta estruturada, proof checker e um
único repair loop por expansão do galho. Falha técnica de grounding não vira
handoff comercial; após a tentativa de reparo, a resposta segura é a pergunta
exata publicada no node de qualificação pendente.

O contrato em Markdown faz parte do fluxo auditável:
`api/contracts/graph-agent-runtime-v3.md` é checksumado dentro de cada
publicação, e `docs/runbooks/graph-agent-runtime-v3-rollout.md` governa
ativação, E2E e rollback.

Exports com nome de persona foram arquivados em 2026-08-19 para
`docs/archive/DEPRECATED_2026-08-19/n8n-legacy-exports/`. Eram legados de
auditoria, nunca importados nem usados pelo provisionador, e confundiam
agentes de IA que os liam como template ativo. O bootstrap local importa
somente os workflows de transporte `whatsapp-*`; workflows de conversa são
criados pela plataforma.

Depois de revisar URLs e credenciais locais, ative manualmente apenas os fluxos
necessários. Nenhum arquivo contém token Meta, telefone, preço, produto, prompt
comercial ou URL de produção. O contrato é sempre
`phone_number_id -> binding` no Brain.
