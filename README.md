# brain-transport

Webhooks Meta/Evolution, normalizacao, mensagens, buffer, midia, dispatch e providers.
Extraido de `brain-plataform` no SHA `b6ee5edc884e233cc0ff41798f4c19239e04fd88`.

Deploy nao executa migrations. Readiness exige schema minimo 131 e `BRAIN_DB_JWT`
com claim `role=brain_transport`; `service_role` e recusada. Endpoints internos
ficam sob `/internal/v1/*` e nao passam pelo gateway publico.
