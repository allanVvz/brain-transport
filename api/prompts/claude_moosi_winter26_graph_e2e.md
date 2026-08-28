# Prompt para Claude: E2E Produto Moosi + Grafo de Brand Util

Voce esta trabalhando no repositorio `ai-brain`. Este prompt continua o trabalho de `prompts/claude_knowledge_graph_validation.md` e deve ser executado com `memory.md` como fonte de verdade da sessao.

## Regra obrigatoria de memoria

1. Antes de qualquer acao, leia `memory.md`.
2. Durante a execucao, registre em `memory.md` decisoes, erros, comandos, resultados e bloqueios.
3. Antes de finalizar, releia `memory.md` e salve o resumo final da sessao.
4. Nunca imprima secrets do `.env`.

## Objetivo desta etapa

Criar um fluxo real e testavel para inserir um novo produto e provar que o grafo de conhecimento da BRAND e util na conversa.

Cenario de negocio solicitado:

- Produto: `Conjunto Moosi com calca Pantalona`
- Aliases/typos esperados em busca: `Conjuto Moosi com calca Pontalona`, `Conijunto Mossi`, `Conjunto Mossi`, `Moosi`
- Preco: `R$ 169,90`
- Cores: 5 cores
- Tamanho: unico
- Campanha: `Campanha Inverno 26`
- Display normalizado da campanha: `Campanha Inverno 2026`
- Relacionados esperados por grafo:
  - `Modais 2026`
  - `Catalogo de Modais 2026`
- Link/catalogo: deve ser configuravel por fixture/env/CLI. Nao hardcodar dominio ou URL no codigo.

O resultado final deve permitir perguntar pelo conjunto via teste de webscraping/chat e obter resposta coerente, com a sidebar de conhecimento exibindo algo equivalente a:

```text
Produto
Conjunto Moosi
R$ 169,90
5 cores
Tamanho unico
<link configurado>

Campanha Inverno 2026

Modais 2026

Catalogo de Modais 2026

Busca por similaridade / relacionados por grafo
<lista de nos relacionados com distancia/path>
```

## Regra de arquitetura nova

Todo produto deve ter preco.

Essa regra nao deve ser um `if product == ...`. Ela deve ser configuravel e auditavel:

- preferencia: usar registry/config de tipo (`knowledge_node_type_registry.config`) ou uma nova tabela de regras de validacao;
- se criar migration nova, nome sugerido: `010_knowledge_product_validation_rules.sql`;
- regra esperada: node/artifact do tipo `product` deve conter preco estruturado em metadata, por exemplo:

```json
{
  "price": {
    "amount": 169.90,
    "currency": "BRL",
    "display": "R$ 169,90"
  }
}
```

Se um produto for adicionado sem preco:

- o classifier/curator deve criar proposta de correcao/rejeicao (`knowledge_curation_proposals`) ou marcar validacao pendente;
- nao deve promover como produto validado silenciosamente.

## Trabalho esperado

### 1. Confirmar base da arquitetura

Antes do novo cenario, rode:

```powershell
python tests/integration_knowledge_curation_architecture.py
python tests/integration_knowledge_curation_architecture.py --require-applied
```

Se `--require-applied` falhar porque a migration 009 ainda nao foi aplicada, pare a parte dependente da 009 e documente o bloqueio. Ainda assim, pode implementar arquivos/rotinas que fiquem prontos para rodar depois da aplicacao.

### 2. Criar fixture parametrizavel do cenario

Criar fixture sugerida:

- `tests/fixtures/knowledge_moosi_winter26.json`

Ela deve conter dados estruturados, nao texto solto apenas:

```json
{
  "persona_slug": "tock-fatal",
  "brand": {
    "slug": "tock-fatal",
    "title": "Tock Fatal"
  },
  "campaign": {
    "slug": "campanha-inverno-2026",
    "title": "Campanha Inverno 2026",
    "aliases": ["Campanha Inverno 26"]
  },
  "product": {
    "slug": "conjunto-moosi-calca-pantalona",
    "title": "Conjunto Moosi com calca Pantalona",
    "aliases": ["Conjuto Moosi com calca Pontalona", "Conijunto Mossi", "Conjunto Mossi", "Moosi"],
    "price": {"amount": 169.90, "currency": "BRL", "display": "R$ 169,90"},
    "colors_count": 5,
    "size": "unico",
    "catalog_url_env": "MOOSI_CATALOG_URL"
  },
  "related": [
    {"node_type": "product", "slug": "modais-2026", "title": "Modais 2026"},
    {"node_type": "faq", "slug": "catalogo-de-modais-2026", "title": "Catalogo de Modais 2026"}
  ],
  "copy": [
    {
      "slug": "copy-conjunto-moosi-inverno-2026",
      "title": "Copy Conjunto Moosi Inverno 2026",
      "body": "Texto comercial curto destacando preco, cores, tamanho unico e campanha."
    }
  ],
  "faq": [
    {
      "slug": "faq-conjunto-moosi-preco",
      "question": "Qual o preco do Conjunto Moosi?",
      "answer_requirements": ["R$ 169,90", "5 cores", "tamanho unico", "link configurado"]
    }
  ],
  "expected_edges": [
    ["conjunto-moosi-calca-pantalona", "part_of_campaign", "campanha-inverno-2026"],
    ["conjunto-moosi-calca-pantalona", "same_topic_as", "modais-2026"],
    ["catalogo-de-modais-2026", "answers_question", "conjunto-moosi-calca-pantalona"],
    ["copy-conjunto-moosi-inverno-2026", "supports_copy", "conjunto-moosi-calca-pantalona"]
  ],
  "expected_sidebar": {
    "product": ["Conjunto Moosi", "R$ 169,90"],
    "campaign": ["Campanha Inverno 2026"],
    "related": ["Modais 2026", "Catalogo de Modais 2026"]
  }
}
```

O teste pode ajustar nomes/campos reais conforme APIs existentes, mas a fixture deve preservar essa intencao.

### 3. Implementar insercao via classifier/curator

Criar teste/rotina sugerida:

- `tests/integration_moosi_winter26_graph.py`

Requisitos:

- parametros:
  - `--scenario tests/fixtures/knowledge_moosi_winter26.json`
  - `--persona-slug`
  - `--catalog-url` ou env `MOOSI_CATALOG_URL`
  - `--dry-run`
  - `--apply`
- no modo `--dry-run`, nao mutar banco; apenas mostrar plano de artifacts/nodes/edges/proposals.
- no modo `--apply`, cadastrar conhecimento via fluxo real do sistema:
  - preferencia: endpoint classifier/curator se existir;
  - fallback: `/knowledge/upload/text` + approve + bootstrap grafo;
  - se o fluxo atual nao suportar metadata estruturado, refatorar o endpoint/servico para suportar.

Validacoes no banco:

- produto existe como `knowledge_artifacts.content_type='product'`;
- produto tem `metadata.price.amount = 169.90` ou estrutura equivalente;
- produto tem node `knowledge_nodes.node_type='product'`;
- campanha tem node `campaign`;
- copy tem artifact/node de `copy`;
- FAQ tem artifact/node de `faq`;
- existem arestas esperadas por `relation_type`;
- nenhum produto validado fica sem preco;
- reexecutar o mesmo scenario nao cria artifact duplicado; deve criar version/proposal ou manter idempotente conforme politica.

### 4. Refatorar ranking por distancia de edges

O resultado da busca/chat-context deve ser guiado pelo grafo, nao por substring.

Implementar ou propor refatoracao em:

- `services/knowledge_graph.py`
- endpoint `/knowledge/chat-context`
- componentes da sidebar em `dashboard/app/messages/page.tsx`

Comportamento esperado:

- dado uma pergunta sobre `Conjunto Moosi`, o resolver identifica o node produto como foco;
- retorna relacionados ordenados por distancia no grafo:
  - distancia 0: produto foco;
  - distancia 1: campanha direta, FAQ direta, copy direta;
  - distancia 1 ou 2: `Modais 2026`, `Catalogo de Modais 2026`, conforme edges;
- cada item relacionado deve expor pelo menos:

```json
{
  "node_id": "...",
  "node_type": "faq",
  "slug": "catalogo-de-modais-2026",
  "title": "Catalogo de Modais 2026",
  "graph_distance": 1,
  "path": ["conjunto-moosi-calca-pantalona", "answers_question", "catalogo-de-modais-2026"],
  "score": 0.0
}
```

Validacao:

- o teste nao deve validar por `contains("Moosi")` apenas;
- deve validar `node_type`, `slug`, `relation_type`, `graph_distance` e `path`;
- a lista de similaridade deve ser exemplo real de traversal/ranking.

### 5. Resposta coerente do chatbot

Criar ou adaptar teste E2E de webscraping:

- sugestao: `tests/e2e_whatsapp_moosi_winter26_graph.py`
- pode reaproveitar `tests/e2e_faq_whatsapp_modal_catalog.py`
- deve aceitar `WA_PROFILE_DIR`, `--bot`, `--contact`, `--question`, `--scenario`, `--catalog-url`

Pergunta sugerida:

```text
Oi, queria saber do conjunto Moosi com calca pantalona. Tem preco, cores e link?
```

Validar por scraping/chat:

- mensagem foi enviada;
- conversa ficou ligada a um lead no banco;
- `/knowledge/chat-context?lead_ref=...&q=...` retorna:
  - produto `conjunto-moosi-calca-pantalona`;
  - preco `169.90`/`R$ 169,90`;
  - campanha `campanha-inverno-2026`;
  - relacionados `modais-2026` e `catalogo-de-modais-2026`;
  - distancias/path de edges;
- resposta do bot contem informacao coerente:
  - preco;
  - cores;
  - tamanho unico;
  - link configurado;
  - campanha ou contexto de inverno, quando disponivel.

Importante:

- Se o WhatsApp/n8n ainda nao estiver usando a resposta local do Brain AI, nao mascarar isso.
- Nesse caso, separar claramente:
  - PASS infraestrutura de dados/grafo/sidebar;
  - FAIL ou BLOCKED resposta final do bot por roteamento n8n.
- Propor/refatorar o caminho correto para que n8n chame `api/routes/process.py` ou para que o sistema local gere a resposta usando `/knowledge/chat-context`.

### 6. Sidebar esperada

Corrigir/validar a sidebar para exibir, a partir do chat-context:

- Produto:
  - `Conjunto Moosi`
  - `R$ 169,90`
  - link configurado
- Campanha:
  - `Campanha Inverno 2026`
- Relacionados:
  - `Modais 2026`
  - `Catalogo de Modais 2026`
- Busca por similaridade:
  - lista ordenada por `graph_distance`/`score`
  - mostrar path ou relacao quando util

Nao criar UI em arvore. A experiencia deve reforcar grafo/relacoes.

### 7. Migracoes/refactors esperados

Propor e implementar migrations apenas se necessarias. Candidatas:

- `010_knowledge_product_validation_rules.sql`
  - adiciona regra configuravel de preco obrigatorio para `product`;
  - pode usar `knowledge_node_type_registry.config` ou tabela nova `knowledge_validation_rules`;
  - adiciona view de produtos sem preco, se util.
- migration para armazenar `graph_distance` nao deve ser feita se distancia puder ser computada em runtime. Preferir runtime/cache a menos que haja necessidade clara.

Refactors provaveis:

- suportar metadata estruturado em upload/classifier;
- normalizar aliases/sinonimos por node metadata;
- melhorar resolver de query para aliases;
- retornar paths/distancias no chat-context;
- renderizar preco/links no card de produto da sidebar.

## Testes obrigatorios

Registrar todos no `memory.md`.

```powershell
python tests/integration_knowledge_curation_architecture.py
python tests/integration_knowledge_curation_architecture.py --require-applied
python tests/integration_moosi_winter26_graph.py --scenario tests/fixtures/knowledge_moosi_winter26.json --dry-run
python tests/integration_moosi_winter26_graph.py --scenario tests/fixtures/knowledge_moosi_winter26.json --apply --catalog-url <URL_CONFIGURADA>
python tests/e2e_whatsapp_moosi_winter26_graph.py --scenario tests/fixtures/knowledge_moosi_winter26.json --catalog-url <URL_CONFIGURADA>
```

Dashboard:

```powershell
cd dashboard
npx.cmd tsc --noEmit
npm.cmd run build
```

Se `npm.cmd run build` falhar com `spawn EPERM`, registrar como bloqueio do ambiente se `npx.cmd tsc --noEmit` passar.

## Criterios de aceite

1. `memory.md` lido no inicio e atualizado no final.
2. Produto Moosi cadastrado sem hardcode na logica.
3. Produto tem preco obrigatorio validado por regra configuravel.
4. Produto, campanha, copy, FAQ e relacionados viram artifacts/nodes/edges.
5. Query sobre o produto retorna resultado por grafo, com `graph_distance` e `path`.
6. Sidebar mostra produto, preco, link, campanha, relacionados e busca por similaridade.
7. Teste E2E por webscraping pergunta pelo produto e valida resposta/chat-context.
8. Se o bot nao responder coerentemente por causa do n8n, o bloqueio fica explicito e o caminho de integracao com `process.py` e proposto/implementado.
9. Reexecutar fixture nao duplica artifact canonico.
10. Nenhum produto validado pode existir sem preco.

## Postura

Atue como arquiteto de dados e produto. O objetivo nao e apenas fazer um teste passar; e criar um grafo de conhecimento de BRAND real e util para venda.

Evite:

- substring como criterio principal;
- acoplamento a Tock Fatal/Modal/Moosi dentro da logica;
- paginas inexistentes na sidebar;
- produto sem preco;
- knowledge item, KB e grafo divergindo como fontes paralelas.
