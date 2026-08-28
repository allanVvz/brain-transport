# AGENTS.md

- Ownership: mensagens, lead_buffer, mídia e status de entrega.
- Não importar código de outro serviço; somente `brain-contracts` em tag exata.
- Persistir inbound antes do 202; indisponibilidade downstream vira retry durável.
- Não executar migrations em deploy e não usar service-role universal.
- Provider `sent`/`delivered` não prova entrega no destino.
