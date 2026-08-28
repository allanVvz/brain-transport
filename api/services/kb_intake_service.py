"""
KB Intake Service — conversational classifier for knowledge ingestion.
Writes to vault → git commit → sync Supabase.
"""
import os
import base64
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

from services import supabase_client
from services import graph_json_importer
from services import graph_document_publisher
from services import knowledge_graph
from services import knowledge_lifecycle
from services import graph_validation
from services import graph_agent_runtime_v3
from services import graph_bundle
from services import graph_bundle_adapter
from services import graph_bundle_error_translations
from services import graph_bundle_publisher
from services import graph_compiler_v3
from schemas.graph_json_v2 import GraphJson
from services.catalog_crawler import crawl_catalog_url
from services.vault_sync import run_sync, VAULT_PATH, ensure_persona_vault_structure, persona_folder_name
from services.model_router import AVAILABLE_MODELS as ROUTER_MODELS
from services.model_router import ModelRouter, ModelRouterError

AVAILABLE_MODELS = {
    **ROUTER_MODELS,
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 - fallback",
}

_GLOBAL_VAULT_CLIENT_FOLDER = "00_GLOBAL"

_CONTENT_TYPE_FOLDERS = {
    "brand":         "01_BRAND",
    "briefing":      "02_BRIEFING",
    "product":       "03_PRODUCTS",
    "product_group": "03_PRODUCT_GROUPS",
    "campaign":      "04_CAMPAIGNS",
    "copy":          "05_COPY",
    "faq":           "06_FAQ",
    "tone":          "07_TONE",
    "audience":      "08_AUDIENCE",
    "competitor":    "09_COMPETITORS",
    "rule":          "10_RULES",
    "offer":         "13_OFFERS",
    "prompt":        "11_PROMPTS",
    "maker_material":"12_MAKER",
    "asset":         "assets",
    "other":         "00_OTHER",
}

_CONTENT_ALIASES = {
    "faq": "faq", "pergunta": "faq", "perguntas": "faq", "kb": "faq",
    "produto": "product", "product": "product",
    "oferta": "offer", "ofertas": "offer", "offer": "offer", "offers": "offer",
    "opcao": "offer", "opção": "offer", "variacao": "offer", "variação": "offer",
    "pacote": "offer", "plano": "offer", "assinatura": "offer", "bundle": "offer", "combo": "offer",
    "product_variant": "offer", "purchase_option": "offer",
    "copy": "copy",
    "campanha": "campaign", "campaign": "campaign",
    "briefing": "briefing",
    "tom": "tone", "tone": "tone",
    "moodboard": "maker_material", "maker": "maker_material",
    "regra": "rule", "regras": "rule", "rule": "rule", "rules": "rule",
}

_PERSONA_ALIASES: dict[str, str] = {}
_PERSONA_DOMAINS: dict[str, str] = {}

_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".mp4", ".pdf", ".ai", ".psd"}
_EVENT_PREVIEW_LIMIT = 280
_EVENT_TRANSCRIPT_MAX_TURNS = 120
_EVENT_CONTEXT_PREVIEW_LIMIT = 400
_BOOTSTRAP_PROMPT = (
    "Use o contexto inicial confirmado pelo operador e a retomada da sessao para continuar "
    "o trabalho imediatamente. Nao cumprimente, nao pergunte 'o que posso fazer por voce hoje' "
    "e nao trate isso como uma conversa vazia. Considere que a segunda tela ja iniciou a conversa. "
    "Responda com o proximo passo util: confirme o que ja entendeu, aponte o que esta pendente, "
    "proponha estrutura de conhecimento se ja houver contexto suficiente e use no maximo 3 perguntas objetivas "
    "somente se faltarem dados criticos."
)

_sessions: dict[str, dict] = {}
_BLOCK_COUNT_KEYS = ("brand", "briefing", "campaign", "audience", "product", "offer", "copy", "faq", "rule", "tone", "asset")
_OFFER_CONTENT_TYPES = {"offer", "product_variant", "purchase_option"}
_INVALID_CRIAR_PERSONAS = {"", "all", "todos", "global"}

AGENT_PROFILES = {
    "sofia": {
        "name": "Sofia",
        "role": "agente de inteligencia marketing comercial",
        "greeting": (
            "Olá! Eu sou a **Sofia**. Aprendi bastante sobre marketing para te ajudar "
            "a construir conhecimento para tua marca."
        ),
    },
    "zaya": {
        "name": "Zaya",
        "role": "agente de marketing visual",
        "greeting": (
            "Olá! Eu sou a **Zaya**. Posso te ajudar a transformar conhecimento "
            "visual em direção criativa para tua marca."
        ),
    },
}


def get_agent_profile(agent_key: str | None = None) -> dict:
    return AGENT_PROFILES.get((agent_key or "sofia").strip().lower(), AGENT_PROFILES["sofia"])


def _load_agent_prompt(name: str, fallback: str) -> str:
    """Load a Sofia agent prompt from the repo-root `agents/` dir at runtime.

    D2: the Sofia Criar/Orquestrar system prompts live in `agents/*.md` and are
    composed at runtime. The inline constant remains the fallback so the backend
    keeps working when the file is absent (fresh checkout / container without the
    agents dir mounted)."""
    try:
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[2] / "agents" / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return fallback


_INLINE_SYSTEM_PROMPT = """Você é uma agente especializada em classificar materiais para a base de conhecimento da plataforma Brain AI.

Sua identidade de conversa vem do estado da sessão. Por padrão, a agente é Sofia, agente de inteligência marketing comercial. Em fluxos futuros, a identidade pode mudar organicamente para Zaya, agente de marketing visual. Nunca se apresente como "Criar"; Criar é o nome da ferramenta/tela, não da agente.

Sua função: conduzir uma conversa objetiva para coletar as informações necessárias de classificação. Seja direto e eficiente. Não utilize mensagens padrão de agradecimento ou explicações sobre o processo técnico de salvamento.

VOCÊ NÃO TEM CAPACIDADE DE SALVAR. Salvar é uma ação exclusiva do operador, executada quando ele clica no botão "Salvar" da interface. Por isso:
- NUNCA diga "salvei", "foi salvo", "salvamento concluído", "estou salvando", "realizando o salvamento" ou frases equivalentes.
- NUNCA simule resultado de salvamento. Não existe IO de gravação no seu lado.
- Após apresentar o `<knowledge_plan>` e obter a confirmação ("sim", "pode", "ok"), apenas finalize com uma frase curta como: "Plano pronto. Clique em **Salvar** para persistir." e marque `"complete": true` no bloco `<classification>`.
- Se o operador perguntar "foi salvo?", responda que o salvamento depende do clique dele no botão Salvar — você não tem essa permissão.

=== MODO GERAR (PRIORIDADE MÁXIMA — SOBREPÕE QUALQUER OUTRA REGRA) ===
Esta seção rege seu comportamento conversacional. Em caso de conflito, ela vence.

GATILHOS DE GERAÇÃO IMEDIATA (não peça mais confirmação, GERE):
- "gere", "gera", "gerar", "pode gerar", "gera agora"
- "sim", "ok", "pode", "manda", "manda ver", "vai", "avança", "continua"
- "cria", "criar", "construa", "monta", "monte", "executa", "executar"
- "estrutura", "estrutura agora", "fecha o plano", "fecha"
Quando QUALQUER um aparecer, você responde com `<knowledge_plan>` completo na MESMA mensagem. Não responda "vou gerar agora" ou "pode confirmar?" — apenas gere.

NÃO RESTRINJA POR content_type INICIAL:
O `content_type` que o operador escolheu na tela sinaliza a INTENÇÃO PRINCIPAL, não limita você a um só nó. Mas você TAMBÉM NÃO infla o plano com nodes que o operador não pediu nem o galho exige. Quando ele pede 1 produto, crie 1 produto.

=== NÃO ALUCINAR PRODUTOS (REGRA FORTE) ===
Termos amplos de campanha/posicionamento NÃO são lista de produtos. Frases como
"óculos esportivos", "moda inverno", "linha premium", "produto feminino",
"coleção nova" descrevem CONTEXTO (campaign/briefing/audience), NÃO produtos.
- NUNCA materialize produtos genéricos a partir desses termos (ex.: NÃO crie 9
  nós chamados "óculos para esportes"). Isso é alucinação.
- Só crie `product` quando houver pelo menos UM destes sinais: nomes reais de
  produtos fornecidos pelo operador; pedido explícito de quantidade ("crie 9
  produtos", "3 produtos por grupo"); instrução "use estes produtos" / "extraia
  do catálogo"; ou catálogo/fonte conectada.
- Se faltarem esses sinais, NÃO invente. Pergunte antes, oferecendo opções, ex.:
  "Entendi a campanha de óculos esportivos. Para montar o grafo sem inventar
  produtos, você quer que eu: A) use produtos já cadastrados; B) crie product
  groups por modelo/coleção; C) aguarde os nomes dos produtos?"

=== PRODUCT_GROUP, COPY E RULE SAO CONTEXTO OPCIONAL ===
Quando o operador pedir grupos explicitamente — "crie 3 grupos de produtos",
"associe 3 produtos para cada grupo", "crie grupos por modelo/coleção", "grupos
Radar, Juliet e HSTN" — `product_group` é OBRIGATÓRIO e estrutural:
- Crie cada `product_group` sob a `audience` (ou campaign/briefing/brand quando
  não houver audience) e pendure cada `product` sob o seu `product_group`.
  Cadeia: audience → product_group → product.
- NUNCA jogue os produtos direto no briefing/audience quando há grupos pedidos.
- `product_group` é OPCIONAL quando o operador NÃO pede grupos: nesse caso o
  product pode ficar direto sob audience/campaign/briefing/brand.
- `copy` e `rule` tambem sao opcionais. Eles adicionam contexto quando existem,
  mas a ausencia deles nao bloqueia a criacao do JSON.
- Antes de salvar, traduza as conexoes para o operador em linguagem concreta,
  por exemplo: "Esse produto deve ficar dentro do grupo de produtos Radar".

=== HIERARQUIA FRACTAL CANÔNICA (ÚNICA VÁLIDA) ===
A árvore principal segue exatamente esta ordem. Pule apenas níveis ausentes — nunca invente node "para preencher".

  persona → brand → briefing → campaign → audience → product_group opcional → product opcional → copy opcional → { faq, gallery }

Cardinalidade primária:
  persona  → brand           (1:1)
  brand    → briefing        (1:1)
  briefing → campaign        (N:N)
  campaign → audience        (N:N)
  audience → product_group   (N:N)
  product_group → product    (N:N, quando product_group existir)
  product  → copy            (N:N, quando copy existir)
  copy     → faq             (1:1)
  copy     → gallery         (1:1)

Asset é camada LATERAL, fora da árvore principal. Asset pode conectar a qualquer node ou a outro asset, sempre via `asset_pending` (não aprovado) ou `asset_approved`. Asset NÃO entra como node da árvore canônica.

Tipos NÃO canônicos (use apenas quando estritamente necessário e marque `status: pendente_validacao`):
  rule, tone, entity, tag. Eles NÃO entram na árvore primária. Se aparecerem, conecte por edge secundária.

Relation types primários (use EXATAMENTE estes no links[]):
  persona_has_brand, brand_has_briefing, briefing_has_campaign, campaign_has_audience,
  audience_has_product_group, product_group_has_product, product_has_copy,
  product_group_has_copy, copy_has_faq, copy_has_gallery

Qualquer edge entre nodes de tipos canônicos que NÃO use uma dessas relations é SECUNDÁRIA (`relation_type: "secondary"`). Edges secundárias podem existir entre quaisquer dois nodes e NÃO definem hierarquia.

=== FAQ É EXPANSÃO DO GALHO, NÃO INVENÇÃO ===
Você não escreve FAQ "pensando o conteúdo". Você chama a tool `generate_faq_from_branch(parent_slug)` quando o operador pedir FAQ. A tool lê o galho ancestral (persona → ... → copy) em tempo real e devolve as perguntas/respostas. Se o operador não pediu FAQ, NÃO crie FAQ por iniciativa própria.

=== GALLERY É APROVAÇÃO, NÃO GERAÇÃO ===
Gallery é destino de assets aprovados. Você nunca gera Gallery por iniciativa: ela aparece quando há copy + asset aprovado pelo operador. Não inclua Gallery em `entries[]` em modo CRIAR.

=== ASSET PENDENTE vs APROVADO ===
Asset criado por você ou subido na sessão entra como pendente. Edge `asset_pending`. Quem aprova é o operador. Você não emite `asset_approved`.

=== CONTRATO DE EXPANSÃO (LEIA COM CALMA) ===
Você NÃO multiplica nodes para "completar" o galho. Não existe mais pacote obrigatório de FAQ, expansão incompleta abstrata ou "1 copy por audience" automática. Cada entry só nasce se:
  (a) o operador pediu explicitamente, OU
  (b) é PRÉ-REQUISITO canônico de uma entry que ele pediu (ex.: criar product_group exige um parent estrutural; product_group só é obrigatório quando o operador pedir grupos).
Quando o pré-requisito não está claro, INFIRA o mínimo e marque `status: pendente_validacao`. NÃO crie ramos paralelos só porque a hierarquia "comportaria mais".

CONEXÕES (parent_slug + links) SÃO OBRIGATÓRIAS:
Toda entry não top-level precisa de:
  (a) `metadata.parent_slug` apontando para o slug do nó pai imediato canônico, OU
  (b) aparecer como `target_slug` em `links[]` com `relation_type` canônico.

REGRAS RÍGIDAS DE ANINHAMENTO (NÃO QUEBRE):
- `product_group` SEMPRE filho de `audience` quando existir audience no plano. NUNCA com `parent_slug="self"` se houver audience.
- `product` fica filho de `product_group` quando esse grupo existir no plano; sem grupo, pode ficar sob audience/campaign/briefing/brand.
- `offer` não é camada obrigatória do galho; use como metadata ou relação secundária quando necessário.
- `copy` fica ligada ao product ou product_group que contextualiza.
- `faq` fica ligada ao card mais específico disponível no branch.

REGRAS DE METADATA OPERACIONAL:
- Quando o operador pedir `metadata.<chave>='<valor>'` (ex.: `test_tag='01'`, `display_price=9`, `flavors_note='consultar sabores'`), PROPAGUE esse metadata EXATAMENTE em TODAS as entries que ele referenciou. NÃO omita por achar que é redundante. NÃO converta o tipo.
- Quando o operador pedir um sufixo padronizado no slug (ex.: `-01`), aplique o sufixo em TODAS as entries que você criar nessa sessão.

USO DE DEFAULTS QUANDO FALTAR DADO:
Se o operador respondeu apenas o público (ex.: "mulheres 30-55 loja física"), use isso para preencher campanha/produto/copy/faq sem nova rodada de perguntas. Marque os campos inferidos com `status: "pendente_validacao"` e adicione `metadata.inferred_from: "operator_hint"`. NÃO trave esperando dado adicional — apenas o conjunto persona+título é absolutamente obrigatório; tudo o mais aceita default.

CONEXÕES (parent_slug + links) SÃO OBRIGATÓRIAS:
Toda entry NÃO top-level (top-level = brand, briefing) precisa de UM dos dois:
  (a) `metadata.parent_slug` apontando para o slug do nó pai imediato, OU
  (b) aparecer como `target_slug` em `links[]` com `relation_type` apropriado.
Sem isso a árvore vira plana e o save é rejeitado pelo validador. NUNCA emita entry sem pai (exceto top-level).

Mapa CANÔNICO de relation_type por par (use SEMPRE estes no `links[]`):
  persona       → persona_has_brand          → brand
  brand         → brand_has_briefing         → briefing
  briefing      → briefing_has_campaign      → campaign
  campaign      → campaign_has_audience      → audience
  audience      → audience_has_product_group → product_group
  product_group → product_group_has_product  → product
  product       → product_has_offer          → offer
  offer         → offer_has_copy             → copy
  copy          → copy_has_faq               → faq
  copy          → copy_has_gallery           → gallery
Qualquer relação fora dessa lista entre nodes canônicos é SECUNDÁRIA (`relation_type: "secondary"`) e não define hierarquia.

RESUMO ANTES DO SAVE:
Após o `<knowledge_plan>`, responda curto, sempre derivado do normalizedPlan:
Status: plano gerado
Resumo: briefing N, público N, produto N, oferta N, copy N, FAQ N, asset N, regra N
Política: árvore piramidal; FAQ por copy; Asset por parent
Pendências bloqueantes: nenhuma
Ação: revisar preview
Se o plano estiver vazio, diga: "Estrutura ainda não gerada." Nunca diga "Plano pronto" sem entries.

NUNCA DECLARE "estruturado" SEM EMITIR `<knowledge_plan>`:
Se você for dizer "o conhecimento está estruturado e pronto para salvar", o `<knowledge_plan>` precisa estar na MESMA mensagem. Caso contrário, o operador não consegue ver/salvar nada e a sessão fica inconsistente.

=== TIPOS DE CONTEÚDO TEXTUAL ===
brand, briefing, product, campaign, copy, faq, tone, audience, competitor, rule, prompt, maker_material, other

=== PARA ASSETS VISUAIS ===
Tipo de asset: background, logo, product, model, banner, story, post, video, icon, other
Função do asset: maker_material, brand_reference, campaign_hero, copy_support, product_showcase, other

=== FLUXO DE CLASSIFICAÇÃO ===
1. Identifique o cliente (obrigatório)
2. Identifique se é asset visual ou conteúdo textual
3. Se asset: pergunte tipo e função
4. Se texto: identifique o tipo de conteúdo
5. Confirme o título (sugira um se não houver)
6. Quando completo, apresente apenas o resumo técnico e aguarde a confirmação de salvamento. NÃO informe que "está realizando o salvamento" ou "agradeço a paciência".

Você consegue extrair múltiplas informações de uma única mensagem. Por exemplo, se o usuário diz "background da marca", você já sabe content_type=asset e asset_type=background; a persona deve vir da sessao ou da confirmacao do operador.

Responda SEMPRE em português. Seja conciso.
NÃO use rótulos como "Classe atual:" ou "Estado:". Inclua apenas o bloco de estado puro no final da mensagem: <classification>{
  "complete": false,
  "persona_slug": null,
  "content_type": null,
  "asset_type": null,
  "asset_function": null,
  "title": null
}
</classification>
Quando TODAS as informações estiverem coletadas E confirmadas pelo usuário, marque "complete": true.
"""

_INLINE_SYSTEM_PROMPT += """

=== FLUXO CAPTURAR / MARKETING GRAPH ===
Quando a sessão trouxer um contexto inicial confirmado pelo operador, leia esse contexto como briefing operacional. Antes de acionar qualquer salvamento, proponha:
1. fontes usadas;
2. entries a criar ou atualizar por nivel hierarquico: brand, campaign, audience, product, variant/color, copy, faq, rule e tone;
3. riscos de invencao e perguntas pendentes.

Para pedidos de copy/marketing, gere propostas hierarquizadas por grafo, não uma lista solta de textos. Exemplo de encadeamento:
brand -> campaign -> audience -> product -> color/variant -> copy -> faq/rule.

Nunca invente preço, cor, disponibilidade, URL, política comercial ou promessa. Use apenas contexto inicial, uploads, mensagens do usuário e conhecimento confirmado. Quando faltar dado, marque como pendente e pergunte ao operador.

=== CRAWLER / SITE COMO EVIDENCIA BRUTA ===
Quando o usuario pedir para ler, coletar ou usar um site, trate o crawler como captura bruta, nao como verdade perfeita.
O crawler pode falhar por HTML inconsistente, JavaScript, imagem, dados duplicados ou dados ausentes.

Se houver resultado do crawler no estado da sessao:
- cite a confianca e os avisos tecnicos;
- use candidatos extraidos como rascunho/evidencia, nao como conhecimento ativo;
- quando preco, cor, condicao comercial, disponibilidade ou atributo estiver ausente, pergunte de forma objetiva ou marque como pendente;
- nao diga "li todos os produtos" se o crawler trouxe confianca baixa/media ou candidatos incompletos;
- proponha uma arvore de conhecimento com status por entry: confirmado, inferido, pendente_validacao.

Ao final da coleta, gere varios conhecimentos, um para cada bloco selecionado pelo operador. Exemplo minimo quando os blocos forem briefing, audience, product, copy e faq:
1. briefing: fonte, escopo, riscos do crawler e regras de validacao;
2. audience: segmentos comerciais, com dores/objetivos/criterios de compra;
3. product: uma entry por produto candidato, usando o titulo do produto quando disponivel. Cor, tamanho, material e preco vao em `metadata` ou `tags` do product, nunca como content_type proprio;
4. copy: copys separadas por publico/canal quando houver informacao suficiente;
5. faq: perguntas e respostas recuperaveis sobre condicoes comerciais, atributos confirmados, uso e objecoes.

Antes de salvar, apresente a lista concreta de entries que serao criadas. Nao finalize com um resumo generico.

=== SAIDA ESTRUTURADA OBRIGATORIA PARA GERACAO ===
Quando o operador pedir "gerar conhecimento", "pode gerar", "criar a arvore" ou equivalente, OU se houver resultados de crawler e blocos selecionados no contexto inicial, OU se a sessao for iniciada com URL e blocos:
- nao responda com resumo generico;
- PRIORIZE gerar o plano imediatamente se houver evidências capturadas;
- gere uma proposta completa em Markdown para leitura humana;
- inclua obrigatoriamente um bloco JSON entre <knowledge_plan> e </knowledge_plan>.
- nao substitua <knowledge_plan> por bloco ```json; o teste E2E e o parser do backend exigem as tags literais.

REGRA CRITICA DE FORMATACAO (NAO QUEBRE):
- ERRADO: ```json\n{...}\n``` (markdown fence)
- ERRADO: JSON solto sem nada ao redor
- CORRETO: <knowledge_plan>\n{...}\n</knowledge_plan>
As tags abertura/fechamento sao OBRIGATORIAS, em letras minusculas, exatamente assim. Nao adicione "json" depois do <knowledge_plan>. Nao envolva em fence. Se voce escrever ```json em vez das tags, o backend cai num fallback inseguro e rejeita o save com "content must be a non-empty string".

O JSON deve seguir este formato:
{
  "source": "URL ou origem",
  "persona_slug": "global",
  "validation_policy": "human_validation_required",
  "entries": [
    {
      "content_type": "brand|briefing|campaign|audience|product_group|product|offer|copy|asset|prompt|faq|maker_material|tone|competitor|rule|entity|other",
      "title": "titulo concreto",
      "slug": "slug-canonico",
      "status": "confirmado|inferido|pendente_validacao",
      "content": "conteudo do conhecimento",
      "tags": ["tag"],
      "metadata": {
        "parent_slug": "slug-do-no-pai"
      }
    }
  ],
  "links": [
    {
      "source_slug": "slug-do-no-pai",
      "target_slug": "slug-do-conhecimento",
      "relation_type": "manual"
    }
  ],
  "missing_questions": []
}

Regras para esse bloco:
- Cada entry deve ter uma ligacao principal. Use `metadata.parent_slug` ou inclua um item em `links`.
- Se nao souber o galho correto, pergunte antes de salvar: brand, briefing/campanha, produto, audiencia ou criar novo galho.
- Sugira o galho a partir de padroes semanticos existentes, mas transforme a decisao em edge principal no JSON.
- Briefings nunca sao soltos: conecte ao produto, audiencia, campanha ou outro no indicado.
- Se ainda nao houver pai melhor, conecte ao menos na persona da sessao.
- precisa conter uma entry para cada bloco selecionado no inicio;
- sempre crie uma estrutura de conhecimento em arvore com multiplos galhos: brand/campaign como raiz quando existirem, audience/product como galhos intermediarios, e copy/faq/rule/asset como folhas;
- evite listas planas: cada entry deve ter titulo, conteudo e contexto suficientes para ficar clara sem depender de relacoes obrigatorias;
- se os blocos incluirem product, gere uma entry por produto conhecido ou candidato;
- se o operador pediu uma quantidade minima, essa quantidade e obrigatoria;
- se o operador pediu 3 produtos e o crawler encontrou so 2, crie o terceiro como produto candidato com status pendente_validacao;
- nao encerre um plano que pediu 3 produtos com apenas 2 products;
- se os blocos incluirem audience, gere publicos concretos, nao "publico geral";
- se os blocos incluirem copy, gere copies concretas e use a ferramenta mental de geracao de copy;
- se os blocos incluirem faq, gere perguntas e respostas recuperaveis, realistas e contextualizadas ao parent direto;
- se o operador pediu FAQ sobre condicoes comerciais, atributos ou objecoes, gere FAQs separadas por parent direto;
- `links` e opcional somente quando todas as entries ja trouxerem `metadata.parent_slug`;
- campos desconhecidos devem ficar como pendente_validacao, nao bloquear a arvore inteira.

=== OUTPUT VALIDATION (HARD CONTRACT) ===
Antes de fechar `<knowledge_plan>`, verifique entrada por entrada:
- `content_type` ESTRITAMENTE in {brand, briefing, campaign, audience, product_group, product, offer, copy, asset, prompt, faq, maker_material, tone, competitor, rule, entity, other}. Qualquer outro valor (incluindo "rules", "publico" ou "category") sera rejeitado pelo banco.
- `title` nao vazio, com pelo menos 3 caracteres.
- `content` nao vazio.
- `tags` deve ser lista de strings (pode ser vazia). Nunca dict.
- `metadata` deve ser objeto JSON (dict). Nunca string ou lista.
- `entries` deve ser lista nao vazia.
Se algum campo nao se encaixar, ajuste a entry — nao gere o plano.

=== BLOCOS SELECIONADOS NA CAPTURA ===
O contexto inicial pode trazer "Blocos de conhecimento solicitados". Trate esses blocos como a intencao inicial do operador, nao como um grafo fixo.

Para cada bloco selecionado, identifique lacunas minimas antes de propor entries:
- brand: nome, posicionamento, promessa, provas e restricoes;
- briefing: objetivo, fonte, escopo, publico e formato de saida;
- campaign: nome, periodo, oferta, publico e produtos relacionados;
- audience: segmento, dores, desejos, objecoes e linguagem;
- product: nome, categoria, beneficios, atributos, preco, cores e disponibilidade;
- (cores, materiais, variantes nao sao bloco proprio: registre como atributo do product correspondente em metadata/tags);
- copy: canal, publico, oferta, tom, CTA e prova;
- faq: pergunta real, resposta confirmada, fonte e produto/campanha ligados;
- rule: politica, condicao, excecao e fonte;
- tone: voz, palavras preferidas, palavras proibidas e exemplos;
- asset: tipo visual, uso, fonte, proporcao e restricoes.

Se durante a conversa o operador pedir outro bloco ou mudar o objetivo, atualize a proposta e pergunte as lacunas desse novo bloco. Nao exija que o operador escreva IDs de grafo como "brand:nome-da-persona"; voce deve transformar respostas naturais em entries atomicas.

=== QUANDO FALTAR INFORMACAO ===
Atencao: o MODO GERAR no topo do prompt sobrepoe esta secao. Aplique-a SOMENTE quando ainda nao houve nenhum gatilho de geracao e voce realmente nao tem dados minimos para construir UMA arvore.

Bloqueadores REAIS (so esses devem travar a geracao):
- persona/cliente: se nao identificado, pergunte;
- titulo canonico: se nao tiver, sugira um a partir da fonte (ex.: "Catalogo principal da colecao").

Para QUALQUER outro campo faltante (preco, cor, disponibilidade, politica, FAQ especifico, etc.) NAO pergunte antes de gerar — preencha com `status: "pendente_validacao"` e adicione na lista `missing_questions[]` do plano. O operador valida depois.

Quando faltar persona OU titulo:
1. "Para continuar preciso confirmar:"
2. Lista numerada curta (no maximo 2 perguntas).
3. Mantenha "complete": false no bloco <classification>.

Apos gerar o plano via <knowledge_plan>, marque "complete": true no <classification> imediatamente. Nao espere mais uma confirmacao.

=== SUGESTOES PROATIVAS ===
Apos a geracao inicial de cards, ofereca proativamente ideias de melhorias ou como aumentar o conhecimento, como:
- "Podemos refinar a descricao de algum produto?"
- "Quer adicionar FAQs sobre politica de troca ou frete?"
- "Que tal criar copys especificas para campanhas de lancamento?"
- "Podemos buscar mais informacoes sobre concorrentes ou publicos-alvo?"

=== CONHECIMENTO DE NEGOCIO ===
- Nao assuma regras comerciais, precos, lotes minimos, trocas ou politicas sem evidencia confirmada na sessao, na fonte ou pelo operador.

=== VISUALIZAÇÃO E ENTREGÁVEIS ===
- Responda em Markdown visualmente rico (use tabelas para preços, negrito para ênfase e listas claras). 
- Suas mensagens serão exibidas em um componente com toggle "View/Code". Capriche na organização do Markdown para que a versão "View" seja elegante e profissional.
- Ao gerar cards de conhecimento (<knowledge_plan>), certifique-se de que cada entrada (regras, faqs, produtos, briefings, públicos) seja uma entry ATÔMICA e DETALHADA.
- Se o operador solicitar um volume alto (ex: 20+ cards), crie uma entry individual para cada FAQ, cada Regra e cada Produto. Não agrupe tudo em um único card de "FAQ Geral" se puder criar 10 cards de FAQ específicos.
"""


_INLINE_SYSTEM_PROMPT += """

=== FAQ EM MODO CRIAR ===
Você não escreve o conteúdo do FAQ. Quando o operador pedir FAQ, emita 1 entry `faq` placeholder com `metadata.parent_slug` apontando para o `copy` correto e `metadata.generate_via=\"branch\"`. O backend chama `generate_faq_from_branch(parent_slug)` ao salvar e preenche perguntas/respostas a partir do galho real. Marque essa entry como `status: pendente_validacao` para passar pela curadoria.

=== CATÁLOGO MULTIPRODUTO ===
Catálogo com várias categorias e dezenas/centenas de produtos: emita 1 product_group por categoria informada (não invente) e 1 product por SKU. Não tente gerar copy/offer/faq automaticamente para cada um — espere o operador pedir o galho que ele quer hoje.
"""

_INLINE_SYSTEM_PROMPT += """

=== CONTRATO CANÔNICO DO MODO CRIAR / SOFIA ===
Esta seção substitui qualquer regra anterior sobre multiplicação automática de FAQ, expansão piramidal forçada ou políticas de count.

Você tem 8 tools determinísticas disponíveis (use-as no tool-loop sempre que estiver ligado):
  - create_node(content_type, title, parent_slug, ...)
  - set_parent(slug, parent_slug)
  - connect_nodes(source_slug, target_slug, relation_type)
  - delete_node(slug)
  - attach_session_asset(parent_slug, reading_index, asset_function, title)
  - validate_plan()
  - find_existing_persona_nodes(types=[...], query="...")
  - generate_faq_from_branch(parent_slug, max_questions=8)

Princípios:
1. Crie SOMENTE o que o operador pediu (mais os pré-requisitos canônicos do galho).
2. Para FAQ, NÃO escreva o conteúdo: chame `generate_faq_from_branch(parent_slug=<slug da copy>)`. A tool lê o galho do grafo e propõe perguntas. Você só insere a entry placeholder.
3. Para Gallery, NÃO crie por iniciativa: ela surge na aprovação de assets.
4. Asset vai como pendente, edge `asset_pending`. Quem aprova é o operador.
5. Antes de criar, sempre rode `find_existing_persona_nodes` para evitar duplicado.
6. Termine sempre com `validate_plan()` antes de fechar a resposta. Se houver violações, conserte e re-valide.

Resumo curto pós-plano:
Status: plano gerado
Blocos: brand N, briefing N, campaign N, audience N, product_group N, product N, offer N, copy N, faq N
Pendências: lista curta ou "nenhuma"
Ação: revisar preview no Curadoria

Se não conseguir montar:
Status: bloqueado
Motivo: faltam dados para X
Ação: responder os campos pendentes (máx 2 perguntas)
"""


# D2: prefer agents/sofia_criar.md when present; fall back to the inline default.
_SYSTEM_PROMPT = _load_agent_prompt("sofia_criar.md", _INLINE_SYSTEM_PROMPT)


def _extract_cls(text: str) -> Optional[dict]:
    match = re.search(r"<classification>(.*?)</classification>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except Exception:
        return None


def _strip_cls(text: str) -> str:
    return re.sub(r"\s*<classification>.*?</classification>", "", text, flags=re.DOTALL).strip()


def _strip_knowledge_plan(text: str) -> str:
    return re.sub(r"\s*<knowledge_plan>.*?</knowledge_plan>", "", text, flags=re.DOTALL).strip()


def _candidate_plan_blocks(text: str) -> list[str]:
    """Yield candidate JSON strings that might contain a knowledge_plan.

    Strategy, in order of trust:
      1. <knowledge_plan>...</knowledge_plan>  (the one true contract)
      2. ```json ... ``` fenced block that contains "entries"
      3. ```...``` fenced block (any language tag) that contains "entries"
      4. Top-level JSON object that contains both "entries" and "persona_slug"

    The model occasionally drops the tags despite the prompt rules. Salvaging
    the output is preferable to failing the save and losing the operator's
    work — but we still log a warning so we know it happened.
    """
    candidates: list[str] = []

    for m in re.finditer(r"<knowledge_plan>\s*(.*?)\s*</knowledge_plan>", text, re.DOTALL):
        candidates.append(m.group(1).strip())

    for m in re.finditer(r"```(?:json|JSON)?\s*\n(.*?)\n```", text, re.DOTALL):
        block = m.group(1).strip()
        if '"entries"' in block:
            candidates.append(block)

    # Last resort: a bare JSON object that walks like a plan.
    if not candidates:
        for m in re.finditer(r"\{[^{}]*\"entries\"[^{}]*\"persona_slug\"[\s\S]*?\}", text):
            candidates.append(m.group(0).strip())
        # Bare object with "entries" only (e.g. when persona_slug is below entries).
        for m in re.finditer(r"\{[\s\S]*?\"entries\"\s*:\s*\[[\s\S]*?\]\s*[\s\S]*?\}", text):
            candidates.append(m.group(0).strip())

    return candidates


def _extract_plan(text: str) -> dict:
    """Extract the knowledge_plan JSON object from a Sofia message."""
    for raw in _candidate_plan_blocks(text):
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    return {}


def _extract_plan_entries(text: str) -> list[dict]:
    """Extrai entradas do knowledge_plan para renderizacao de cards no chat."""
    plan = _extract_plan(text)
    entries = plan.get("entries") if isinstance(plan, dict) else None
    return entries if isinstance(entries, list) else []


def count_blocks_by_type(entries: list[dict] | None) -> dict[str, int]:
    counts = {key: 0 for key in _BLOCK_COUNT_KEYS}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        ctype = _normalize_content_type_alias(entry.get("content_type"))
        if ctype in counts:
            counts[ctype] += 1
        elif ctype in _OFFER_CONTENT_TYPES:
            counts["offer"] += 1
    return counts


def _normalize_content_type_alias(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return _CONTENT_ALIASES.get(raw, raw)


def _invalid_criar_persona(persona_slug: str | None) -> bool:
    return str(persona_slug or "").strip().lower() in _INVALID_CRIAR_PERSONAS


def _normalize_block_counts(value: Any) -> dict[str, int]:
    counts = {key: 0 for key in _BLOCK_COUNT_KEYS}
    if not isinstance(value, dict):
        return counts
    for key in counts:
        try:
            counts[key] = max(0, int(value.get(key) or 0))
        except Exception:
            counts[key] = 0
    return counts


def _counts_from_plan_or_initial(plan: Optional[dict], initial_counts: Optional[dict] = None) -> dict[str, int]:
    entries = plan.get("entries") if isinstance(plan, dict) else None
    if isinstance(entries, list) and entries:
        return count_blocks_by_type(entries)
    return _normalize_block_counts(initial_counts)


def _format_block_counts(counts: Optional[dict[str, Any]]) -> str:
    normalized = _normalize_block_counts(counts)
    return ", ".join(f"{key} {value}" for key, value in normalized.items() if value) or "sem blocos"


def _build_live_memory_summary(session: dict, plan: Optional[dict] = None, *, last_change: str = "") -> str:
    persona = session.get("persona_slug") or (session.get("classification") or {}).get("persona_slug") or "nao informada"
    source = session.get("source_url") or (((session.get("mission_state") or {}).get("source") or {}).get("url")) or "nao informada"
    initial_counts = _format_block_counts(session.get("initial_block_counts"))
    current_counts = _format_block_counts(session.get("current_block_counts"))
    tree_mode = (plan or session.get("knowledge_plan") or {}).get("tree_mode") if isinstance(plan or session.get("knowledge_plan"), dict) else None
    branch_policy = (plan or session.get("knowledge_plan") or {}).get("branch_policy") if isinstance(plan or session.get("knowledge_plan"), dict) else None
    lines = [
        f"Persona global da sessao: {persona}.",
        f"Fonte principal: {source}.",
        f"Plano inicial: {initial_counts}.",
        f"Plano atual: {current_counts}.",
        f"Politica de arvore: {branch_policy or 'top_down_pyramidal'}.",
        f"Modo da arvore: {tree_mode or 'pyramidal'}.",
        "Nao salvar usando o plano inicial se o plano atual foi expandido.",
    ]
    if last_change:
        lines.append(f"Ultima alteracao do operador/agente: {last_change}.")
    return "\n".join(lines)


def _count_mismatch_message(expected: dict[str, int], actual: dict[str, int]) -> str | None:
    for key in _BLOCK_COUNT_KEYS:
        exp = int(expected.get(key) or 0)
        got = int(actual.get(key) or 0)
        if exp > 0 and got != exp:
            label = "FAQ" if key == "faq" else key
            return f"Plan mismatch: current plan has {exp} {label} but save payload has {got} {label}."
    return None


def _explicit_total_count_requested(session: Optional[dict], block_id: str) -> bool:
    text = _session_text_for_branch_policy(session or {})
    label = "faq" if block_id == "faq" else "asset"
    return bool(re.search(rf"\b{label}s?\s+(?:total|no total|totais)\b|\b(?:usar|use)\s+\d+\s+{label}s?\s+no\s+total\b", text, re.I))


def _normalize_count_policy(plan: dict, session: Optional[dict], block_id: str) -> str:
    key = f"{block_id}_count_policy"
    raw = str(plan.get(key) or "").strip().lower()
    if block_id == "faq" and raw in {"grouped", "single_grouped", "grouped_markdown"}:
        return "grouped"
    if block_id == "faq" and raw in {"per_branch", "golden_dataset_per_branch"}:
        return "per_branch"
    if raw == "total" or _explicit_total_count_requested(session, block_id):
        return "total"
    if block_id == "faq":
        return "grouped"
    return "per_parent"


def _direct_parent_type_for(plan: dict, block_id: str) -> str:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    types = {_entry_type(entry) for entry in entries}
    if block_id == "faq":
        return str(plan.get("faq_parent_type") or ("copy" if "copy" in types else "product" if "product" in types else "product_group" if "product_group" in types else "audience")).strip() or "copy"
    if block_id == "asset":
        return str(plan.get("asset_parent_type") or ("product" if "product" in types else "campaign" if "campaign" in types else "briefing")).strip() or "product"
    return "product"


def _direct_parents_for(plan: dict, block_id: str) -> list[dict]:
    parent_type = _direct_parent_type_for(plan, block_id)
    return [
        entry for entry in (plan.get("entries") or [])
        if isinstance(entry, dict) and _entry_type(entry) == parent_type and entry.get("slug")
    ]


def _entry_count_under_parents(plan: dict, child_type: str, parents: list[dict]) -> int:
    parent_slugs = {str(parent.get("slug")) for parent in parents if parent.get("slug")}
    return sum(
        1 for entry in (plan.get("entries") or [])
        if isinstance(entry, dict)
        and _entry_type(entry) == child_type
        and (_entry_parent_slug(entry) or "") in parent_slugs
    )


def _faq_golden_dataset_questions_by_slug(plan: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in (plan.get("entries") or []):
        if not isinstance(entry, dict) or _entry_type(entry) != "faq":
            continue
        meta = _entry_metadata(entry)
        slug = str(entry.get("slug") or "")
        if slug:
            out[slug] = max(0, int(meta.get("question_count") or 0))
    return out


def _faq_golden_dataset_question_total(plan: dict) -> int:
    return sum(_faq_golden_dataset_questions_by_slug(plan).values())


def _expansion_summary(plan: dict, session: Optional[dict] = None) -> dict[str, dict[str, Any]]:
    faq_policy = _normalize_count_policy(plan, session, "faq")
    asset_policy = _normalize_count_policy(plan, session, "asset")
    faq_per_parent = max(0, int(plan.get("faq_count_per_parent") or 1))
    asset_per_parent = max(0, int(plan.get("asset_count_per_parent") or _requested_variation_count(session or {}, "asset", 0)))
    faq_parents = _direct_parents_for(plan, "faq") if faq_policy in {"per_parent", "per_branch"} else []
    asset_parents = _direct_parents_for(plan, "asset") if asset_policy == "per_parent" else []
    faq_created = count_blocks_by_type(plan.get("entries") or []).get("faq", 0)
    asset_created = count_blocks_by_type(plan.get("entries") or []).get("asset", 0)
    faq_questions = _faq_golden_dataset_question_total(plan)
    faq_expected = (
        max(0, int(plan.get("faq_total_count") or _requested_variation_count(session or {}, "faq", faq_created)))
        if faq_policy == "total"
        else 1 if faq_policy == "grouped" and (faq_created > 0 or _requested_variation_count(session or {}, "faq", 0) > 0)
        else len(faq_parents)
    )
    asset_expected = (
        max(0, int(plan.get("asset_total_count") or _requested_variation_count(session or {}, "asset", asset_created)))
        if asset_policy == "total"
        else len(asset_parents) * asset_per_parent
    )
    return {
        "faq": {
            "count_policy": faq_policy,
            "parent_type": _direct_parent_type_for(plan, "faq"),
            "count_per_parent": faq_per_parent,
            "configured": faq_per_parent if faq_policy == "per_parent" else faq_expected,
            "expected": faq_expected,
            "created": faq_created,
            "terminal_branches": len(faq_parents),
            "questions_total": faq_questions,
            "questions_per_document": _faq_golden_dataset_questions_by_slug(plan),
        },
        "asset": {
            "count_policy": asset_policy,
            "parent_type": _direct_parent_type_for(plan, "asset"),
            "count_per_parent": asset_per_parent,
            "configured": asset_per_parent if asset_policy == "per_parent" else asset_expected,
            "expected": asset_expected,
            "created": asset_created,
        },
    }


def summarize_normalized_plan(plan: dict) -> dict[str, Any]:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    counts = count_blocks_by_type(entries)
    return {
        "entry_count": len(entries),
        "current_block_counts": counts,
        "link_count": len(plan.get("links") or []),
        "tree_mode": plan.get("tree_mode") or "pyramidal",
        "branch_policy": plan.get("branch_policy") or "top_down_pyramidal",
        "faq_count_policy": plan.get("faq_count_policy") or "grouped",
        "faq_parent_type": plan.get("faq_parent_type") or "copy",
        "asset_count_policy": plan.get("asset_count_policy") or "per_parent",
        "copy_policy": plan.get("copy_policy") or "per_product_context",
        "expansion": _expansion_summary(plan),
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _plan_hash(plan: dict) -> str:
    canonical = {
        key: value
        for key, value in (plan or {}).items()
        if key not in {"summary", "validation", "plan_hash"}
    }
    return hashlib.sha256(_stable_json(canonical).encode("utf-8")).hexdigest()


def normalized_plan_to_graph_json(plan: dict, session: dict) -> GraphJson:
    persona_slug = str(
        plan.get("persona_slug")
        or (session.get("classification") or {}).get("persona_slug")
        or session.get("persona_slug")
        or ""
    ).strip()
    if not persona_slug:
        raise ValueError("persona_slug is required to build graph_json")
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    graph_id = f"kb-intake:{session.get('id') or uuid.uuid4().hex}:{_plan_hash(plan)[:16]}"
    persona_id = f"persona:{persona_slug}"
    source = plan.get("source") or (session.get("classification") or {}).get("source") or "pending_source"
    nodes: list[dict[str, Any]] = [
        {
            "id": persona_id,
            "node_type": "persona",
            "slug": persona_slug,
            "label": persona_slug,
            "data": {"status": "validated", "source": source},
        }
    ]
    slug_to_node_id: dict[str, str] = {persona_slug: persona_id, "self": persona_id}
    entry_by_slug: dict[str, dict] = {}
    for entry in entries:
        slug = str(entry.get("slug") or _slug_for_plan_entry(entry.get("title") or entry.get("content_type") or "node"))
        ctype = _entry_type(entry)
        if not slug or ctype == "persona":
            continue
        slug_to_node_id[slug] = f"{ctype}:{slug}"
        entry_by_slug[slug] = entry

    def branch_path_for(entry: dict, node_id: str) -> list[str]:
        path = [node_id]
        cursor = entry
        seen: set[str] = set()
        for _ in range(30):
            parent_slug = str(_entry_parent_slug(cursor) or "self")
            if parent_slug in seen:
                break
            seen.add(parent_slug)
            parent_id = slug_to_node_id.get(parent_slug) or persona_id
            path.append(parent_id)
            if parent_id == persona_id:
                break
            cursor = entry_by_slug.get(parent_slug) or {}
            if not cursor:
                break
        return list(reversed(path))

    for entry in entries:
        ctype = _entry_type(entry)
        if ctype == "persona":
            continue
        slug = str(entry.get("slug") or _slug_for_plan_entry(entry.get("title") or ctype))
        node_id = slug_to_node_id.get(slug) or f"{ctype}:{slug}"
        parent_slug = str(_entry_parent_slug(entry) or "self")
        parent_id = slug_to_node_id.get(parent_slug) or persona_id
        metadata = dict(entry.get("metadata") or {})
        content = str(entry.get("content") or metadata.get("markdown") or "")
        status = str(entry.get("status") or metadata.get("validation_status") or "pending_validation")
        data = {
            **metadata,
            "source": source,
            "status": status,
            "validation_status": metadata.get("validation_status") or status,
            "content": content,
            "markdown": metadata.get("markdown") or content,
            "tags": entry.get("tags") or [],
            "parent_slug": parent_slug,
        }
        if ctype == "faq":
            branch_path = branch_path_for(entry, node_id)
            direct_parent = next((node for node in nodes if node["id"] == parent_id), None)
            data.setdefault("markdown_document", True)
            data.setdefault("question_count", max(1, len(re.findall(r"(?m)^###\s+", content))))
            data.setdefault("branch_path", branch_path)
            data.setdefault("source_node_id", parent_id)
            data.setdefault("source_node_type", (direct_parent or {}).get("node_type") or "unknown")
        nodes.append(
            {
                "id": node_id,
                "node_type": ctype,
                "slug": slug,
                "label": str(entry.get("title") or slug),
                "parent_id": parent_id,
                "data": data,
            }
        )

    node_type_by_id = {node["id"]: node["node_type"] for node in nodes}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for node in nodes:
        parent_id = node.get("parent_id")
        if not parent_id:
            continue
        pair = (str(parent_id), str(node["id"]))
        if pair in seen_edges:
            continue
        seen_edges.add(pair)
        parent_type = node_type_by_id.get(str(parent_id), "")
        child_type = str(node.get("node_type") or "")
        relation = graph_json_importer.RELATION_BY_PAIR.get((parent_type, child_type), "main")
        edges.append(
            {
                "id": f"edge:{parent_id}->{node['id']}",
                "source": parent_id,
                "target": node["id"],
                "relation": relation,
                "primary_tree": True,
                "metadata": {"primary_tree": True, "source": "kb_intake.normalized_plan"},
            }
        )

    return GraphJson.model_validate(
        {
            "schema_version": "2.0",
            "graph_id": graph_id,
            "tenant": "local",
            "persona_slug": persona_slug,
            "brand_slug": next((entry.get("slug") for entry in entries if _entry_type(entry) == "brand"), None),
            "status": "draft",
            "nodes": nodes,
            "edges": edges,
            "validation": {"is_valid": True, "errors": []},
        }
    )


def _active_workflow_binding(persona_id: str) -> dict:
    """Same routing switch api.routes.personas._active_binding reads --
    duplicated locally (4 lines) rather than importing a route module from a
    service module."""
    if not persona_id:
        return {}
    for binding in supabase_client.get_workflow_bindings(persona_id):
        if binding.get("active"):
            return binding
    return {}


def _persona_uses_graph_bundle_pipeline(persona_id: str) -> bool:
    return graph_agent_runtime_v3.binding_uses_v3(_active_workflow_binding(persona_id))


def _current_persona_base_bundle(persona_id: str, persona_slug: str) -> dict | None:
    """Compile the persona's live graph into a base bundle + compiled
    document, so a new Sofia plan can be diffed/staged additively instead of
    overwriting the persona's existing knowledge. Mirrors the manual process
    used to activate the Tock Fatal two-brand graph this session. Returns
    None if the persona has no live graph yet (first-ever bundle for it)."""
    persona = supabase_client.get_persona_by_id(persona_id)
    if not persona:
        return None
    node_rows, edge_rows = supabase_client.list_all_knowledge_graph(
        persona_id=persona_id, limit_nodes=10000
    )
    if not node_rows:
        return None
    id_by_projection: dict[str, str] = {}
    base_nodes: list[dict[str, Any]] = []
    for row in node_rows:
        metadata = dict(row.get("metadata") or {})
        bundle_node_id = str(
            metadata.pop("graph_json_node_id", None)
            or f"{row.get('node_type')}:{row.get('slug')}"
        )
        metadata.pop("graph_bundle_draft_checksum", None)
        id_by_projection[str(row.get("id"))] = bundle_node_id
        base_nodes.append({
            "id": bundle_node_id,
            "node_type": str(row.get("node_type") or ""),
            "slug": str(row.get("slug") or ""),
            "title": str(row.get("title") or ""),
            "summary": str(row.get("summary") or ""),
            "tags": list(row.get("tags") or []),
            "status": str(row.get("status") or "validated"),
            "projection_node_id": str(row.get("id")),
            "data": metadata,
        })
    base_edges: list[dict[str, Any]] = []
    for row in edge_rows:
        metadata = dict(row.get("metadata") or {})
        if metadata.get("active", True) is False:
            continue
        source_id = id_by_projection.get(str(row.get("source_node_id")))
        target_id = id_by_projection.get(str(row.get("target_node_id")))
        if not source_id or not target_id:
            continue
        edge_id = str(metadata.pop("graph_json_edge_id", None) or row.get("id"))
        metadata.pop("graph_bundle_draft_checksum", None)
        metadata.pop("primary_tree", None)
        base_edges.append({
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "relation_type": str(row.get("relation_type") or "contains"),
            "weight": float(row.get("weight") or 1.0),
            "metadata": metadata,
        })
    return {"nodes": base_nodes, "edges": base_edges}


def _save_via_graph_bundle(
    session: dict,
    session_id: str,
    plan_payload: dict,
    plan_state: dict,
    plan_warnings: list[dict],
) -> dict:
    """The v3 counterpart to the graph_json_v2_store save path above: builds
    a GraphBundle from the confirmed plan entries, computes a
    PublicationPlan, and stops there for explicit human approval instead of
    auto-publishing -- staging/activation only happen via a follow-up call to
    POST /kb-intake/{session_id}/approve-publication with the exact
    checksums shown here (see api/routes/kb_intake.py)."""
    persona_id = str(session.get("persona_id") or "").strip()
    persona_slug = str(session.get("persona_slug") or "").strip()
    try:
        if not persona_id or not persona_slug:
            raise ValueError("session is missing persona_id/persona_slug")
        base_bundle = _current_persona_base_bundle(persona_id, persona_slug)
        adapted = graph_bundle_adapter.normalized_plan_to_graph_bundle(
            plan_payload, session, base_bundle=base_bundle,
        )
        bundle = graph_bundle_adapter.ensure_branch_reachability(adapted["bundle"])
        held_back = adapted["held_back"]
        current_document = None
        if base_bundle is not None:
            persona = supabase_client.get_persona_by_id(persona_id)
            node_rows, edge_rows = supabase_client.list_all_knowledge_graph(
                persona_id=persona_id, limit_nodes=10000
            )
            current_document = graph_compiler_v3.compile_graph(
                persona=persona, node_rows=node_rows, edge_rows=edge_rows,
                embedding_profile=bundle["metadata"]["embedding_profile"],
            )
        plan = graph_bundle.build_publication_plan(
            bundle, current_document=current_document, next_version=1,
        )
    except Exception as exc:
        response = {
            "error": "Nao foi possivel montar o GraphBundle a partir do plano da Sofia.",
            "error_code": "GRAPH_BUNDLE_BUILD_FAILED",
            "errors": [str(exc)],
            "plan_state": plan_state,
        }
        _emit_kb_event(
            "kb_intake_dialog_rejected", session=session, source="kb-intake.save",
            status="rejected", transcript=True, result=response,
        )
        return response

    if plan.get("validation_errors"):
        response = {
            "error": "O GraphBundle ainda nao pode ser publicado -- corrija os itens abaixo.",
            "error_code": "GRAPH_BUNDLE_VALIDATION_FAILED",
            "requires_sofia_intervention": True,
            "graph_bundle_validation": {
                "blocking": plan["validation_errors"],
                "translated": graph_bundle_error_translations.translate_errors(plan["validation_errors"]),
            },
            "held_back": held_back,
            "plan_state": plan_state,
        }
        _emit_kb_event(
            "kb_intake_dialog_rejected", session=session, source="kb-intake.save",
            status="rejected", transcript=True, result=response,
        )
        return response

    session["stage"] = "awaiting_publication_approval"
    session["status"] = "pending_approval"
    session["pending_graph_bundle"] = bundle
    session["pending_publication_plan"] = plan
    _save_session(session)
    completion_payload = {
        "status": "pending_approval",
        "success": True,
        "warnings": plan_warnings,
        "held_back": held_back,
        "publication_plan": plan,
        "plan_state": plan_state,
        "plan_hash": plan_state.get("plan_hash"),
        "approval_instructions": (
            "Publicacao nao foi ativada. Revise publication_plan (nodes/edges "
            "adicionados, breaking_contract_changes) e chame "
            "POST /kb-intake/{session_id}/approve-publication com "
            "draft_checksum e runtime_checksum exatamente como aparecem aqui."
        ),
    }
    _emit_kb_event(
        "kb_intake_dialog_completed", session=session, source="kb-intake.save",
        status="completed", transcript=True, result=completion_payload,
    )
    return {
        "ok": True,
        "success": True,
        "status": "pending_approval",
        "warnings": plan_warnings,
        "held_back": held_back,
        "publication_plan": plan,
        "plan_state": plan_state,
        "plan_hash": plan_state.get("plan_hash"),
        "approval_instructions": completion_payload["approval_instructions"],
    }


def _plan_validation(violations: list[str] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    blocking = [str(item) for item in (violations or []) if str(item).strip()]
    return {
        "valid": len(blocking) == 0,
        "blocking_violations": blocking,
        "warnings": [str(item) for item in (warnings or []) if str(item).strip()],
    }


def _leaf_alert_warnings(plan: dict, session: Optional[dict] = None) -> list[str]:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    counts = count_blocks_by_type(entries)
    child_parent_slugs = {_entry_parent_slug(entry) for entry in entries if _entry_parent_slug(entry)}
    link_sources = {
        str(link.get("source_slug"))
        for link in (plan.get("links") or [])
        if isinstance(link, dict) and link.get("source_slug") and link.get("target_slug")
    }
    warnings: list[str] = []
    if session and _rule_required(session, plan) and counts.get("rule", 0) == 0:
        warnings.append("pending_rule: regra comercial explicita ainda nao foi criada.")
    if session and _offers_required(session, plan) and counts.get("offer", 0) == 0:
        warnings.append("pending_offer: oferta/condicao comercial ainda nao foi criada.")
    if session and _requested_variation_count(session, "asset", 0) > 0 and counts.get("asset", 0) == 0:
        warnings.append("pending_asset: asset solicitado ainda nao foi criado.")
    if counts.get("product", 0) > 0 and counts.get("copy", 0) == 0:
        warnings.append("pending_copy: produtos ainda nao tem copy. Deseja gerar agora?")
    if (counts.get("product", 0) > 0 or counts.get("product_group", 0) > 0) and counts.get("faq", 0) == 0:
        warnings.append("pending_faq: ainda nao ha FAQ comercial. Deseja gerar agora?")
    if counts.get("faq", 0) == 0:
        warnings.append("pending_embedded: Embedded sera conectado depois que houver FAQ aprovada.")
    elif counts.get("embedded", 0) == 0:
        warnings.append("pending_embedded_connection: FAQ existe, mas ainda nao ha conexao final com Embedded.")
    # FAQ is terminal-valid in the plan: after human approval it is connected
    # to the persona's Embedded node automatically, so the plan itself must not
    # treat a pending FAQ as a structural error.
    terminal_ok = {"persona", "asset", "embedded", "gallery", "faq"}
    for entry in entries:
        slug = str(entry.get("slug") or "").strip()
        ctype = _entry_type(entry)
        if not slug or ctype in terminal_ok:
            continue
        has_child = slug in child_parent_slugs or slug in link_sources
        if has_child:
            continue
        title = str(entry.get("title") or slug).strip()
        if ctype == "rule":
            warnings.append(
                f"pending_terminal_connection: O node RULE ficou sem saida: {title}. Deseja conectar RULE antes do FAQ, transformar em orientacao global da campanha ou manter como pendencia?"
            )
        else:
            warnings.append(
                f"pending_terminal_connection: O node {ctype or 'desconhecido'} ficou sem saida: {title}. Deseja conectar, transformar em orientacao global ou manter como pendencia?"
            )
    return warnings


def _plan_state_from_normalized(plan: dict, session: Optional[dict] = None, *, violations: Optional[list[str]] = None) -> dict[str, Any]:
    normalized_plan = dict(plan or {})
    summary = summarize_normalized_plan(normalized_plan)
    normalized_plan["summary"] = summary
    resolved_violations = violations if violations is not None else validate_sofia_knowledge_plan(normalized_plan, session=session)
    validation = _plan_validation(resolved_violations, _leaf_alert_warnings(normalized_plan, session))
    graph_json_payload: dict[str, Any] | None = None
    if validation.get("valid"):
        try:
            graph_doc = normalized_plan_to_graph_json(normalized_plan, session or {})
            graph_json_payload = graph_doc.model_dump()
        except Exception as exc:
            validation = _plan_validation([f"graph_json_generation_failed: {exc}"], validation.get("warnings") or [])
    plan_hash = _plan_hash(normalized_plan)
    diagnostic = build_plan_diagnostic(normalized_plan, session, validation["blocking_violations"]) if validation["blocking_violations"] else None
    state: dict[str, Any] = {
        "normalized_plan": normalized_plan,
        "validation": validation,
        "summary": summary,
        "plan_hash": plan_hash,
    }
    if graph_json_payload is not None:
        state["graph_json"] = graph_json_payload
    if diagnostic is not None:
        state["diagnostic"] = diagnostic
    return state


def normalize_validate_summarize_plan(raw_plan: dict, session: dict, *, live_edit: bool = False) -> dict[str, Any]:
    effective_session = session
    if live_edit and isinstance(raw_plan, dict):
        effective_session = dict(session or {})
        effective_session["current_block_counts"] = count_blocks_by_type(raw_plan.get("entries") or [])
    normalized_plan = _normalize_sofia_knowledge_plan(raw_plan, effective_session)
    plan_state = _plan_state_from_normalized(normalized_plan, session=effective_session)
    return plan_state


def _store_plan_state(session: dict, plan_state: dict[str, Any], *, last_change: str = "") -> None:
    normalized_plan = plan_state.get("normalized_plan") or {}
    summary = plan_state.get("summary") or summarize_normalized_plan(normalized_plan)
    validation = plan_state.get("validation") or _plan_validation()
    plan_hash = str(plan_state.get("plan_hash") or _plan_hash(normalized_plan))
    session["normalized_plan"] = normalized_plan
    session["knowledge_plan"] = normalized_plan
    session["last_proposed_plan"] = normalized_plan
    session["plan_validation"] = validation
    session["plan_summary"] = summary
    session["knowledge_plan_summary"] = summary
    session["plan_hash"] = plan_hash
    if plan_state.get("graph_json"):
        session["graph_json"] = plan_state.get("graph_json")
    session["current_block_counts"] = summary.get("current_block_counts") or count_blocks_by_type(normalized_plan.get("entries") or [])
    session["plan_changed"] = True
    session["memory_summary"] = _build_live_memory_summary(session, normalized_plan, last_change=last_change)


# Top-level node_types that may be a tree root without an explicit parent.
# Everything else MUST connect to one of these (transitively) via parent_slug
# or links[]. Keeps the operator's "no isolated node" rule enforceable.
SOFIA_TOP_LEVEL_TYPES: frozenset[str] = frozenset({"persona", "brand", "briefing"})

# Preferred parent node_types per child type. When Sofia emits an entry
# without parent_slug, _auto_infer_parent_slugs walks this list and picks
# the FIRST matching entry already declared in the plan (most recent of
# that type). This mirrors the architectural intent: faq belongs to a
# product, products belong to a campaign, copies belong to a product, etc.
# Top-down chain enforced by the operator's hierarchy:
#   Persona → Brand → Campaign|Briefing → Audience → Product Group → Product → Copy|FAQ|Asset
# Each child's preferred parents are listed from CLOSEST to fallback. The
# audience/product_group pivot prevents flat shortcuts that bypass the semantic
# grouping step.
_PREFERRED_PARENT_TYPES: dict[str, tuple[str, ...]] = {
    "briefing": ("brand",),
    "campaign": ("briefing", "brand"),
    "audience": ("campaign", "briefing", "brand"),
    "product_group": ("audience",),
    # Product prefers product_group. Falls back only for older plans that did
    # not request or emit the grouping layer.
    "product": ("product_group",),
    "offer": ("product",),
    "entity": ("product", "audience", "campaign", "briefing", "brand"),
    "tone": ("brand", "briefing", "campaign"),
    "rule": ("campaign", "briefing", "brand"),
    "competitor": ("brand", "briefing"),
    # Per-product children prefer the product directly. Falling back to
    # audience preserves the semantic step instead of jumping to campaign.
    "copy": ("product", "product_group", "campaign", "audience"),
    "faq": ("copy", "product", "product_group", "audience", "campaign", "briefing", "brand"),
    "asset": ("product", "audience", "campaign", "brand"),
    "maker_material": ("product", "campaign", "brand"),
    "prompt": ("campaign", "brand", "briefing"),
    "other": ("product", "audience", "campaign", "brand", "briefing"),
}


def _shared_slug_tokens(a: str, b: str) -> set[str]:
    """Return tokens shared between two slugs, excluding generic content-type
    prefixes/suffixes (faq, copy, briefing, etc.). Single-digit tokens like
    '1' or '2' are kept because they distinguish "Produto A" from "Kit
    Modal 2" in slugs like 'faq-preco-produto-b'."""
    if not a or not b:
        return set()
    blacklist = {
        "faq", "copy", "produto", "product", "audiencia", "audience",
        "campanha", "campaign", "brand", "briefing", "rule", "regra",
        "tone", "tom", "asset", "ativo",
        # Generic positional/connector words.
        "para", "pra", "com", "sem", "do", "da", "de", "e", "a", "o",
    }
    tokens_a = {t for t in (a or "").lower().split("-") if t and t not in blacklist}
    tokens_b = {t for t in (b or "").lower().split("-") if t and t not in blacklist}
    return tokens_a & tokens_b


def _best_parent_by_slug(orphan: dict, candidates: list[dict]) -> Optional[dict]:
    """Pick the candidate whose slug shares the most non-generic tokens with
    the orphan's slug/title. Returns None when there is no signal at all.

    This enables per-product FAQ/copy/asset matching: an entry slugged
    `faq-preco-produto-a` correctly attaches to the product
    `produto-a-cores` instead of an unrelated product earlier in the plan.
    """
    if not candidates:
        return None
    orphan_slug = (orphan.get("slug") or "")
    orphan_title = (orphan.get("title") or "")
    orphan_blob = f"{orphan_slug} {orphan_title}".lower()
    best = None
    best_score = 0
    for cand in candidates:
        if cand is orphan:
            continue
        cand_slug = (cand.get("slug") or "")
        cand_title = (cand.get("title") or "")
        # Substring match on full slug = strong signal.
        if cand_slug and cand_slug in orphan_blob:
            return cand
        shared = _shared_slug_tokens(orphan_slug, cand_slug)
        shared |= _shared_slug_tokens(orphan_title, cand_title)
        if len(shared) > best_score:
            best_score = len(shared)
            best = cand
    return best if best_score >= 1 else None


def _auto_infer_parent_slugs(plan: dict) -> int:
    """Backstop hierarchy: when Sofia forgets parent_slug for non-top-level
    entries, infer one from the surrounding semantic order. Mutates the plan
    in place. Returns the number of entries that received an inferred parent.

    Algorithm (per orphan entry):
      1. Skip if the entry is top-level (brand, briefing, persona).
      2. Skip if metadata.parent_slug is already set, or the entry's slug
         appears as a target in plan.links (explicit parent already declared).
      3. Walk _PREFERRED_PARENT_TYPES[ctype] in order. For each preferred
         parent type, prefer the candidate whose slug/title shares tokens
         with the orphan (per-product matching). Fall back to MOST RECENT
         only when no slug-similar candidate exists. This keeps each
         product's FAQ/copy attached to its OWN product instead of
         collapsing into a single most-recent product.
      4. If no preferred match, fall back to the first top-level entry
         (brand/briefing) anywhere in the plan.
      5. If still no candidate, the entry stays orphan; the validator will
         reject the plan with a precise message.
    """
    if not isinstance(plan, dict):
        return 0
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        return 0

    # Pre-compute link targets so explicit links aren't overwritten.
    raw_links = plan.get("links")
    link_targets: set[str] = set()
    if isinstance(raw_links, list):
        for link in raw_links:
            if isinstance(link, dict) and link.get("target_slug"):
                link_targets.add(str(link["target_slug"]))

    # Index existing entries by lowercase content_type → list (preserve order).
    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ctype = (entry.get("content_type") or "").lower()
        if ctype:
            by_type.setdefault(ctype, []).append(entry)

    inferred_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ctype = (entry.get("content_type") or "").lower()
        if not ctype or ctype in SOFIA_TOP_LEVEL_TYPES:
            continue
        slug = entry.get("slug")
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if metadata.get("parent_slug"):
            continue
        if slug and str(slug) in link_targets:
            continue

        best: Optional[dict] = None
        for parent_type in _PREFERRED_PARENT_TYPES.get(ctype, ("brand", "briefing")):
            candidates = [c for c in (by_type.get(parent_type) or []) if c is not entry and c.get("slug")]
            if not candidates:
                continue
            # Try slug-similarity first so each FAQ/copy attaches to its OWN
            # parent product when there are multiple products in the plan.
            picked = _best_parent_by_slug(entry, candidates)
            if picked is None:
                # Fallback: most recent of that type.
                picked = candidates[-1]
            best = picked
            break

        # Fallback: first top-level entry anywhere in plan. Do not use this
        # for commercial outputs: if a copy/FAQ cannot find product/copy
        # context, the plan is ambiguous and must ask before saving.
        if best is None and ctype not in {"copy", "faq"}:
            for candidate in entries:
                if not isinstance(candidate, dict) or not candidate.get("slug"):
                    continue
                if (candidate.get("content_type") or "").lower() in SOFIA_TOP_LEVEL_TYPES:
                    best = candidate
                    break
        if best is None or best is entry:
            continue

        if not isinstance(entry.get("metadata"), dict):
            entry["metadata"] = {}
        entry["metadata"]["parent_slug"] = str(best.get("slug"))
        entry["metadata"].setdefault("parent_inferred", True)
        entry["metadata"].setdefault(
            "parent_inferred_from",
            (best.get("content_type") or "unknown"),
        )
        inferred_count += 1
    return inferred_count


def validate_sofia_knowledge_plan(plan: dict, session: Optional[dict] = None) -> list[str]:
    """Validate a Sofia <knowledge_plan> JSON against the DB contract.

    Returns a list of human-readable violations (empty list = valid). Mirrors the
    constraints enforced by knowledge_items (NOT NULL, CHECK content_type, types)
    AND the architectural rule "no isolated node": every non-top-level entry
    needs an explicit parent (metadata.parent_slug or appearance as
    links[*].target_slug).
    """
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]

    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["plan.entries must be a non-empty list"]

    raw_links = plan.get("links")
    links: list[dict] = raw_links if isinstance(raw_links, list) else []
    if raw_links is not None and not isinstance(raw_links, list):
        errors.append("plan.links must be a list when present")

    # Build set of slugs that are referenced as link targets (i.e. have a parent
    # via the links[] array). Each link must carry source_slug + target_slug.
    target_slugs_with_parent: set[str] = set()
    declared_slugs: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("slug"):
            declared_slugs.add(str(entry["slug"]))
    for lidx, link in enumerate(links):
        if not isinstance(link, dict):
            errors.append(f"links[{lidx}] must be a JSON object")
            continue
        src = link.get("source_slug")
        tgt = link.get("target_slug")
        if not src or not tgt:
            errors.append(f"links[{lidx}] requires source_slug and target_slug")
            continue
        target_slugs_with_parent.add(str(tgt))

    allowed = supabase_client.KNOWLEDGE_ITEM_CONTENT_TYPES
    for idx, entry in enumerate(entries):
        prefix = f"entry[{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be a JSON object")
            continue

        content_type = _normalize_content_type_alias(entry.get("content_type"))
        if not content_type:
            errors.append(f"{prefix} missing content_type")
        elif content_type not in allowed:
            errors.append(
                f"{prefix} content_type {content_type!r} not allowed "
                f"(expected one of {sorted(allowed)})"
            )

        title = entry.get("title")
        if not isinstance(title, str) or len(title.strip()) < 3:
            errors.append(f"{prefix} title must be a string of at least 3 chars")

        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{prefix} content must be a non-empty string")

        tags = entry.get("tags")
        if tags is not None and not isinstance(tags, list):
            errors.append(f"{prefix} tags must be a list, got {type(tags).__name__}")

        metadata = entry.get("metadata") or {}
        if entry.get("metadata") is not None and not isinstance(entry.get("metadata"), dict):
            errors.append(f"{prefix} metadata must be a dict, got {type(entry.get('metadata')).__name__}")
            metadata = {}

        # Hierarchical contract: non-top-level entries need an explicit parent.
        ctype_lower = (content_type or "").lower()
        if ctype_lower and ctype_lower not in SOFIA_TOP_LEVEL_TYPES:
            slug = entry.get("slug")
            has_parent_slug = bool(metadata.get("parent_slug"))
            has_link_target = bool(slug) and str(slug) in target_slugs_with_parent
            if not has_parent_slug and not has_link_target:
                errors.append(
                    f"{prefix} content_type {ctype_lower!r} requires a parent "
                    f"(set metadata.parent_slug OR add an entry to links[] with target_slug={slug!r})"
                )

    slug_to_entry = {
        str(entry.get("slug")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("slug")
    }
    product_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "product"]
    product_group_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "product_group"]
    audience_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "audience"]
    faq_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "faq"]
    asset_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "asset"]
    offer_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "offer"]
    copy_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "copy"]
    rule_entries = [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "rule"]

    slugs = [str(entry.get("slug")) for entry in entries if isinstance(entry, dict) and entry.get("slug")]
    if len(slugs) != len(set(slugs)):
        errors.append("plan.entries must not contain duplicate slugs")

    # Preview/create validation is intentionally tolerant for complementary
    # commercial layers. Missing offer/rule/copy/FAQ/asset/embedded are surfaced
    # as Sofia suggestions in _leaf_alert_warnings instead of blocking the
    # primary tree. Structural errors below still block.

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        ctype_lower = _entry_type(entry)
        parent_slug = _entry_parent_slug(entry)
        parent_entry = slug_to_entry.get(parent_slug or "")
        parent_type = _entry_type(parent_entry or {})
        if ctype_lower in {"tag", "knowledge_item", "kb_entry", "mention"}:
            errors.append(f"entry[{idx}] {ctype_lower} cannot be a primary tree card")
        if _sofia_tools_enabled():
            # Canonical validator — parent rules sourced from the SHARED
            # graph_validation module so the Create and Graph paths can never
            # diverge. product_group is an OPTIONAL layer; product may hang off a
            # group OR directly off audience/campaign/briefing/brand.
            if parent_slug not in ("self", None, "") and ctype_lower in graph_validation.CANONICAL_PARENTS:
                violation = graph_validation.parent_violation(ctype_lower, parent_type)
                if violation:
                    errors.append(f"entry[{idx}] {violation} (via slug {parent_slug!r})")
                available_types = {
                    _entry_type(candidate)
                    for candidate in entries
                    if isinstance(candidate, dict)
                }
                contextual = graph_validation.contextual_parent_violation(ctype_lower, parent_type, available_types)
                if contextual:
                    errors.append(f"entry[{idx}] {contextual} (via slug {parent_slug!r})")
        else:
            # Legacy validator preserved for non-canonical callers.
            if ctype_lower == "audience" and parent_slug and parent_type not in {"campaign", "briefing", "brand", ""}:
                errors.append(f"entry[{idx}] audience must stay under campaign/briefing/brand, got parent {parent_slug!r}")
            if ctype_lower == "product":
                # product_group is a legal grouping layer between audience and
                # product. Allowed parents come from the SHARED graph_validation
                # module (same rule the Graph path uses).
                allowed_product_parents = graph_validation.canonical_parents("product") | {""}
                if parent_slug and parent_type not in allowed_product_parents:
                    errors.append(f"entry[{idx}] product has invalid parent {parent_slug!r}")
                elif audience_entries and not product_group_entries and parent_type not in {"audience", ""}:
                    errors.append(f"entry[{idx}] product must stay under audience when audience exists")
                if "audience" in str(entry.get("slug") or "").lower():
                    errors.append(f"entry[{idx}] product slug must not embed audience slug")
            if ctype_lower == "offer" and parent_type != "product":
                errors.append(f"entry[{idx}] offer must stay under product, got parent {parent_slug!r}")
            if ctype_lower == "copy":
                allowed_copy_parents = {"product", "product_group", "campaign", "audience", "briefing", "brand", ""}
                if parent_type not in allowed_copy_parents:
                    errors.append(f"entry[{idx}] copy has invalid parent {parent_slug!r}")
            if ctype_lower == "faq":
                allowed_faq_parents = {"copy", "product", "product_group", "audience", "campaign", "briefing", "brand"}
                if parent_type not in allowed_faq_parents:
                    errors.append(f"entry[{idx}] faq has invalid parent {parent_slug!r}")
            if ctype_lower == "rule" and parent_type not in {"campaign", "briefing", "brand", "persona", ""}:
                errors.append(f"entry[{idx}] rule must stay under campaign/briefing/brand, got parent {parent_slug!r}")

    tree_mode = str(plan.get("tree_mode") or "pyramidal").strip() or "pyramidal"
    if tree_mode == "single_branch" and not _technical_product_faq_requested(session or {}):
        copy_slugs_by_product: dict[str, set[str]] = {}
        for copy in [entry for entry in entries if isinstance(entry, dict) and _entry_type(entry) == "copy"]:
            product_slug = _entry_parent_slug(copy)
            if product_slug:
                copy_slugs_by_product.setdefault(product_slug, set()).add(str(copy.get("slug") or ""))
        for idx, faq in enumerate(faq_entries):
            parent_slug = _entry_parent_slug(faq)
            parent_entry = slug_to_entry.get(parent_slug or "")
            if _entry_type(parent_entry or {}) == "product" and copy_slugs_by_product.get(parent_slug or ""):
                errors.append(
                    f"entry[{idx}] faq must use copy parent in single_branch when product {parent_slug!r} has copy"
                )

    parent_by_child = {
        str(entry.get("slug")): _entry_parent_slug(entry)
        for entry in entries
        if isinstance(entry, dict) and entry.get("slug")
    }
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("slug"):
            continue
        ctype = _entry_type(entry)
        if ctype in SOFIA_TOP_LEVEL_TYPES:
            continue
        current = str(entry.get("slug"))
        seen: set[str] = set()
        reaches_persona = False
        for _ in range(64):
            parent = parent_by_child.get(current)
            if not parent:
                break
            if parent == "self":
                reaches_persona = True
                break
            if parent in seen:
                errors.append(f"entry[{idx}] cycle detected in parent chain")
                break
            seen.add(parent)
            current = parent
        if not reaches_persona and ctype not in {"copy", "faq", "offer", "rule", "asset"}:
            errors.append(f"entry[{idx}] has no complete path to persona")

    faq_policy = str(plan.get("faq_count_policy") or "per_branch").strip() or "per_branch"
    if session and faq_policy == "total":
        requested_total = _requested_variation_count(session or {}, "faq", 8)
        if requested_total >= 0 and len(faq_entries) > requested_total:
            errors.append(
                f"faq_count_policy total allows at most {requested_total} FAQs without confirmation (found {len(faq_entries)})"
            )

    expansion = _expansion_summary(plan, session)
    faq_expansion = expansion.get("faq") or {}
    asset_expansion = expansion.get("asset") or {}

    if len(entries) > 1 and not (plan.get("links") or []):
        errors.append("plan.links must not be empty when the hierarchy already contains clear parent/child relations")

    return errors


# --------------------------------------------------------------------------- #
# Plan diagnostic — agrupador de violacoes + grafo visual                     #
# --------------------------------------------------------------------------- #
#
# When the planner emits blocking_violations, the UI used to show a long flat
# list. build_plan_diagnostic() classifies every entry (valid / error / cycle
# / orphan), groups violations by root cause and produces markdown questions
# that the operator can answer to repair the plan. The dashboard renders this
# structure as a colored graph; the backend keeps the data source of truth.

_DIAGNOSTIC_KIND_TITLES = {
    "cycle": "Ciclo na cadeia de pais",
    "no_path_to_persona": "Caminho incompleto ate a persona",
    "offer_under_product": "Oferta com parent invalido",
    "audience_parent": "Publico com parent invalido",
    "product_under_audience": "Produto fora de audience",
    "product_invalid_parent": "Produto com parent invalido",
    "copy_parent": "Copy com parent invalido",
    "faq_parent": "FAQ com parent invalido",
    "rule_parent": "Regra com parent invalido",
    "invalid_primary_tree_type": "Tipo nao permitido na arvore principal",
    "duplicate_slug": "Slugs duplicados no plano",
    "missing_parent_slug": "Entrada sem parent obrigatorio",
    "faq_expansion_incomplete": "FAQ esperado incompleto",
    "asset_expansion_incomplete": "Asset expansion incompleto",
    "offer_missing": "Oferta exigida e ausente",
    "rule_missing": "Regra comercial exigida e ausente",
    "links_missing": "Plano sem links explicitos",
    "other": "Outras pendencias",
}

_DIAGNOSTIC_KIND_DESCRIPTIONS = {
    "cycle": "Uma ou mais entradas apontam para parent que volta a propria entrada ou para um descendente, formando ciclo. A arvore deixa de ter raiz valida.",
    "no_path_to_persona": "Existem entradas cuja cadeia de pais nao termina na persona. Sao galhos soltos no grafo.",
    "offer_under_product": "Oferta precisa ficar diretamente abaixo de um produto. Encontramos oferta(s) com parent diferente.",
    "audience_parent": "Audience deve ficar abaixo de campaign, briefing ou brand.",
    "product_under_audience": "Quando ha audience no plano e nenhum product_group, todo produto deve ficar abaixo da audience.",
    "product_invalid_parent": "Produto deve ficar abaixo de product_group, audience, campaign, briefing ou brand.",
    "copy_parent": "Copy deve ficar ligada ao produto, grupo de produtos, campanha ou publico que ela contextualiza.",
    "faq_parent": "FAQ deve ficar ligada ao card mais especifico do galho: copy, produto, grupo de produtos, publico, campanha, briefing ou brand.",
    "rule_parent": "Regra comercial deve ficar abaixo de campaign, briefing, brand ou persona.",
    "invalid_primary_tree_type": "Tipos auxiliares (tag, mention, knowledge_item, kb_entry) nao podem estar na arvore principal.",
    "duplicate_slug": "O mesmo slug aparece em mais de uma entrada do plano.",
    "missing_parent_slug": "Entrada sem persona/brand/briefing precisa declarar metadata.parent_slug ou aparecer em links[].",
    "faq_expansion_incomplete": "O numero de FAQs gerados ficou abaixo do esperado pela politica de expansao.",
    "asset_expansion_incomplete": "O numero de assets gerados ficou abaixo do esperado pela politica de expansao.",
    "offer_missing": "A sessao indica preco, kit, plano ou variacao comercial mas nenhuma oferta foi criada.",
    "rule_missing": "A sessao indica regras comerciais (prazo, troca, pagamento) mas nenhuma regra foi criada.",
    "links_missing": "Plano tem mais de uma entrada mas nao declarou links explicitos entre elas.",
    "other": "Outras pendencias reportadas pelo validador.",
}

_DIAGNOSTIC_KIND_REPAIRS = {
    "cycle": "Reabra as entradas afetadas e corrija metadata.parent_slug para apontar para um node real acima na hierarquia (persona/brand/briefing/campaign/audience/product).",
    "no_path_to_persona": "Conecte o galho ate a persona reescrevendo metadata.parent_slug das entradas afetadas (a cadeia precisa chegar em persona/self).",
    "offer_under_product": "Escolha o produto correto como parent da oferta (metadata.parent_slug aponta para um slug de product).",
    "audience_parent": "Aponte o audience para campaign, briefing ou brand existente.",
    "product_under_audience": "Coloque os produtos abaixo da audience (ou de um product_group sob ela).",
    "product_invalid_parent": "Reaponte os produtos para product_group, audience, campaign, briefing ou brand.",
    "copy_parent": "Reaponte as copies para o produto/contexto comercial; use oferta como parent apenas quando o operador pedir copy por oferta.",
    "faq_parent": "Reaponte o FAQ para a copy, produto, grupo de produtos ou outro card especifico que ele responde.",
    "rule_parent": "Aponte as regras para campaign, briefing, brand ou persona.",
    "invalid_primary_tree_type": "Remova esses tipos da arvore principal; tags/mentions ficam em camada auxiliar.",
    "duplicate_slug": "Renomeie um dos slugs duplicados.",
    "missing_parent_slug": "Declare metadata.parent_slug ou adicione a entrada em links[].",
    "faq_expansion_incomplete": "Confirme a politica de expansao FAQ ou aumente o numero de copies/produtos terminais.",
    "asset_expansion_incomplete": "Confirme a politica de assets ou ajuste count_per_parent.",
    "offer_missing": "Adicione pelo menos uma oferta abaixo de cada produto.",
    "rule_missing": "Adicione pelo menos uma regra abaixo de campaign/briefing/brand.",
    "links_missing": "Declare os links pai/filho em plan.links para refletir a hierarquia.",
    "other": "Revise as mensagens cruas listadas para entender a pendencia.",
}


def _diagnostic_classify_violation(message: str) -> tuple[str, Optional[int]]:
    """Map a raw violation string to (kind, entry_index)."""
    if not isinstance(message, str):
        return "other", None
    idx_match = re.match(r"entry\[(\d+)\]\s+(.*)", message)
    affected_idx = int(idx_match.group(1)) if idx_match else None
    if "cycle detected in parent chain" in message:
        return "cycle", affected_idx
    if "has no complete path to persona" in message:
        return "no_path_to_persona", affected_idx
    if "offer must stay under product" in message:
        return "offer_under_product", affected_idx
    if "audience must stay under" in message:
        return "audience_parent", affected_idx
    if "product must stay under audience" in message:
        return "product_under_audience", affected_idx
    if "product has invalid parent" in message or "product slug must not embed audience slug" in message:
        return "product_invalid_parent", affected_idx
    if "copy has invalid parent" in message or "copy precisa ficar ligada" in message:
        return "copy_parent", affected_idx
    if "faq has invalid parent" in message or "faq must stay under copy" in message or "faq must use copy parent" in message:
        return "faq_parent", affected_idx
    if "rule must stay under" in message:
        return "rule_parent", affected_idx
    if "cannot be a primary tree card" in message:
        return "invalid_primary_tree_type", affected_idx
    if "duplicate slugs" in message:
        return "duplicate_slug", None
    if "requires a parent" in message:
        return "missing_parent_slug", affected_idx
    if "FAQ expansion incomplete" in message or "FAQs esperado incompleto" in message:
        return "faq_expansion_incomplete", None
    if "Asset expansion incomplete" in message:
        return "asset_expansion_incomplete", None
    if "offer required" in message:
        return "offer_missing", None
    if "rule required" in message:
        return "rule_missing", None
    if "links must not be empty" in message:
        return "links_missing", None
    return "other", affected_idx


def _diagnostic_expected_parents(ctype: str, *, has_audience: bool, has_offer: bool, has_copy: bool) -> list[str]:
    if ctype in {"persona", "brand", "briefing"}:
        return []
    if ctype == "campaign":
        return ["brand", "briefing", "persona"]
    if ctype == "audience":
        return ["campaign", "briefing", "brand"]
    if ctype == "product":
        if has_audience:
            return ["product_group", "audience"]
        return ["product_group", "audience", "campaign", "briefing", "brand"]
    if ctype == "offer":
        return ["product"]
    if ctype == "copy":
        return ["product", "product_group", "campaign", "audience"]
    if ctype == "faq":
        return ["copy", "product", "product_group", "audience", "campaign", "briefing", "brand"]
    if ctype == "rule":
        return ["campaign", "briefing", "brand", "persona"]
    if ctype == "tone":
        return ["brand", "briefing", "persona"]
    if ctype == "asset":
        return ["product", "product_group", "copy", "faq", "brand", "campaign", "audience"]
    return []


def _diagnostic_walk_parent_chain(
    start_slug: str,
    parent_by_child: dict[str, Optional[str]],
) -> tuple[bool, bool, list[str]]:
    """Return (reaches_persona, has_cycle, visited_path)."""
    current = start_slug
    seen: set[str] = set()
    path: list[str] = []
    for _ in range(64):
        parent = parent_by_child.get(current)
        if not parent:
            return False, False, path
        if parent == "self":
            return True, False, path
        if parent == current or parent in seen:
            return False, True, path
        seen.add(parent)
        path.append(parent)
        current = parent
    return False, True, path


def _diagnostic_entry_title(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    title = entry.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    slug = entry.get("slug")
    return str(slug).strip() if slug else ""


def _diagnostic_questions_markdown(
    plan: dict,
    nodes: list[dict],
    root_causes: list[dict],
    by_kind: dict[str, dict],
) -> str:
    questions: list[str] = []

    offer_problems = [n for n in nodes if n["type"] == "offer" and n["status"] in {"error", "orphan", "cycle"}]
    product_nodes = [n for n in nodes if n["type"] == "product" and n.get("slug")]
    for offer in offer_problems:
        product_options = ", ".join(f"`{p['slug']}`" for p in product_nodes) or "nenhum produto disponivel"
        questions.append(
            f"A oferta `entry[{offer['entry_index']}]` (slug `{offer['slug'] or '?'}`) pertence a qual produto? "
            f"Opcoes: {product_options}."
        )

    if "no_path_to_persona" in by_kind or "cycle" in by_kind:
        orphan_or_cycle = [n for n in nodes if n["status"] in {"orphan", "cycle"}]
        if orphan_or_cycle:
            questions.append(
                "As entradas com ciclo ou sem caminho ate a persona devem ser reorganizadas como "
                "`persona -> brand -> briefing -> campaign -> audience -> product_group -> product -> copy -> faq`?"
            )

    if "offer_under_product" in by_kind:
        questions.append(
            "As ofertas devem ser distribuidas entre os produtos existentes ou todas pertencem a um unico produto?"
        )

    if "copy_parent" in by_kind:
        questions.append("As copies devem contextualizar qual produto, grupo de produtos, campanha ou publico?")

    if "faq_parent" in by_kind or "faq_expansion_incomplete" in by_kind:
        questions.append("Esse FAQ responde sobre qual copy, produto ou grupo de produtos?")

    campaign_slugs = [n["slug"] for n in nodes if n["type"] == "campaign" and n.get("slug")]
    if campaign_slugs and ("offer_under_product" in by_kind or "no_path_to_persona" in by_kind):
        questions.append(
            f"A(s) campanha(s) {', '.join('`' + s + '`' for s in campaign_slugs)} devem ser apenas contexto geral acima dos produtos?"
        )

    questions.append(
        "Deseja que o sistema regenere o plano automaticamente seguindo a arvore "
        "`briefing -> campaign -> audience -> product_group -> product -> copy -> faq`?"
    )

    if not questions:
        return ""

    lines = ["## Perguntas para corrigir o plano", ""]
    for idx, q in enumerate(questions, 1):
        lines.append(f"{idx}. {q}")
    return "\n".join(lines)


def _diagnostic_persona_existing(session: Optional[dict]) -> dict[str, list[dict[str, str]]]:
    """Surface existing persona nodes (campaign/audience/product/asset/faq/copy/offer/rule)
    so Sofia questions can offer concrete options keyed to real slugs/titles instead of
    asking the operator to invent answers."""
    if not isinstance(session, dict):
        return {}
    persona_context = session.get("persona_context")
    if not isinstance(persona_context, dict):
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for ntype, rows in persona_context.items():
        if not isinstance(rows, list):
            continue
        bucket: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            slug = row.get("slug")
            if not slug:
                continue
            bucket.append({
                "slug": str(slug),
                "title": str(row.get("title") or slug),
            })
        if bucket:
            out[str(ntype)] = bucket
    return out


def _diagnostic_humanize_type(ctype: Optional[str]) -> str:
    table = {
        "persona": "persona",
        "brand": "marca",
        "briefing": "briefing",
        "campaign": "campanha",
        "audience": "publico",
        "product": "produto",
        "offer": "oferta",
        "copy": "copy",
        "faq": "FAQ",
        "rule": "regra comercial",
        "asset": "asset visual",
        "tone": "tom de voz",
    }
    return table.get(str(ctype or ""), str(ctype or "entrada"))


def _diagnostic_entry_label(node: Optional[dict]) -> str:
    if not isinstance(node, dict):
        return "entrada"
    title = (node.get("title") or "").strip()
    if title:
        return title
    slug = (node.get("slug") or "").strip()
    if slug:
        return slug
    return f"entry[{node.get('entry_index')}]"


def _sofia_options_for_parent_choice(
    affected_node: dict,
    plan_nodes: list[dict],
    existing: dict[str, list[dict[str, str]]],
    expected_types: list[str],
) -> list[dict[str, Any]]:
    """Build options listing concrete candidate parents in the plan and persona
    snapshot. Always closes with "manter pendente"."""
    options: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for ptype in expected_types:
        # Candidates already in the plan
        for cand in plan_nodes:
            if cand["type"] != ptype:
                continue
            slug = cand.get("slug") or ""
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            options.append({
                "label": f"Conectar em {_diagnostic_humanize_type(ptype)} {cand.get('title') or slug}",
                "action": "set_parent_slug",
                "payload": {
                    "entry_index": affected_node.get("entry_index"),
                    "parent_slug": slug,
                    "parent_type": ptype,
                    "source": "plan",
                },
            })
        # Candidates from persona snapshot
        for cand in existing.get(ptype, []):
            slug = cand.get("slug") or ""
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            options.append({
                "label": f"Conectar em {_diagnostic_humanize_type(ptype)} existente \"{cand.get('title') or slug}\"",
                "action": "set_parent_slug",
                "payload": {
                    "entry_index": affected_node.get("entry_index"),
                    "parent_slug": slug,
                    "parent_type": ptype,
                    "source": "persona",
                },
            })
    options.append({
        "label": "Manter como pendencia para validar depois",
        "action": "mark_pending",
        "payload": {"entry_index": affected_node.get("entry_index")},
    })
    return options


def _sofia_questions_from_diagnostic(
    plan: dict,
    session: Optional[dict],
    nodes: list[dict],
    by_kind: dict[str, dict],
) -> list[dict[str, Any]]:
    """Translate technical blocking violations into operator-facing questions.

    Each returned entry contains: kind, technical_error, human_summary,
    probable_cause, question, options[], severity, plus optional affected node
    metadata. The list is intended to be the primary surface in the diagnostic
    modal; raw `root_causes[].raw_messages` stays available for debugging."""
    existing = _diagnostic_persona_existing(session)
    nodes_by_index: dict[int, dict] = {int(n.get("entry_index", -1)): n for n in nodes if isinstance(n, dict)}
    questions: list[dict[str, Any]] = []

    for kind, bucket in by_kind.items():
        raw_messages: list[str] = list(bucket.get("raw_messages") or [])
        affected_idxs: list[int] = list(bucket.get("affected_entry_indexes") or [])

        if kind == "no_path_to_persona":
            for idx in affected_idxs:
                node = nodes_by_index.get(idx)
                if not node:
                    continue
                expected = node.get("expected_parent_types") or []
                ntype = node.get("type") or ""
                label = _diagnostic_entry_label(node)
                if ntype == "rule":
                    human = f"A regra comercial \"{label}\" ficou fora do caminho principal da persona."
                    cause = "A regra nao tem parent declarado entre campaign, briefing, brand ou persona."
                    question = f"Onde devo aplicar a regra \"{label}\"?"
                elif ntype == "faq":
                    human = f"O FAQ \"{label}\" ficou sem conexao clara com a campanha confirmada."
                    cause = "FAQ precisa pertencer ao card mais especifico que ele responde: copy, produto, grupo de produtos, publico, campanha, briefing ou brand."
                    question = f"Esse FAQ deve responder sobre qual parte da campanha?"
                elif ntype == "copy":
                    human = f"A copy \"{label}\" ficou sem destino comercial."
                    cause = "Copy precisa apontar para produto, grupo de produtos, campanha ou publico."
                    question = f"Em qual contexto a copy \"{label}\" deve aparecer?"
                elif ntype == "offer":
                    human = f"A oferta \"{label}\" nao esta ligada a um produto."
                    cause = "Toda oferta precisa de um produto pai."
                    question = f"A oferta \"{label}\" pertence a qual produto?"
                elif ntype == "asset":
                    human = f"O asset \"{label}\" nao esta conectado em nenhum lugar do plano."
                    cause = "Asset precisa pertencer a produto, oferta, copy, FAQ, campanha ou audience."
                    question = f"Em qual node devo conectar o asset \"{label}\"?"
                else:
                    human = f"A entrada \"{label}\" ({_diagnostic_humanize_type(ntype)}) ficou fora do caminho principal."
                    cause = "A cadeia de pais nao chega ate a persona."
                    question = f"Em qual node a entrada \"{label}\" deve ficar conectada?"
                tech = next(
                    (m for m in raw_messages if isinstance(m, str) and f"entry[{idx}]" in m),
                    raw_messages[0] if raw_messages else "",
                )
                questions.append({
                    "kind": kind,
                    "technical_error": tech,
                    "affected_entry_index": idx,
                    "affected_title": label,
                    "affected_type": ntype,
                    "severity": "blocking",
                    "human_summary": human,
                    "probable_cause": cause,
                    "question": question,
                    "options": _sofia_options_for_parent_choice(node, nodes, existing, expected),
                })
            continue

        if kind == "asset_expansion_incomplete":
            expansion = _expansion_summary(plan, session) if isinstance(plan, dict) else {}
            asset_info = expansion.get("asset") or {}
            expected_count = int(asset_info.get("expected") or 0)
            created_count = int(asset_info.get("created") or 0)
            existing_assets = existing.get("asset", [])
            options: list[dict[str, Any]] = []
            for cand in existing_assets[:4]:
                options.append({
                    "label": f"Conectar asset existente \"{cand.get('title') or cand['slug']}\" a campanha",
                    "action": "attach_existing_asset",
                    "payload": {"slug": cand["slug"], "scope": "campaign"},
                })
                options.append({
                    "label": f"Conectar asset existente \"{cand.get('title') or cand['slug']}\" ao produto",
                    "action": "attach_existing_asset",
                    "payload": {"slug": cand["slug"], "scope": "product"},
                })
            options.append({
                "label": "Enviar um novo asset agora",
                "action": "upload_asset",
                "payload": {},
            })
            options.append({
                "label": "Seguir sem asset neste plano",
                "action": "drop_asset_requirement",
                "payload": {},
            })
            human = (
                f"O plano esperava {expected_count} asset visual"
                + ("(s)" if expected_count != 1 else "")
                + f", mas {created_count} foi/foram conectado(s)."
            )
            questions.append({
                "kind": kind,
                "technical_error": raw_messages[0] if raw_messages else "",
                "severity": "blocking",
                "human_summary": "A Sofia esperava um asset visual para este plano, mas nenhum asset foi conectado.",
                "probable_cause": human,
                "question": "Quer conectar o asset visual existente a campanha ou seguir sem asset?",
                "options": options,
            })
            continue

        if kind == "faq_expansion_incomplete":
            expansion = _expansion_summary(plan, session) if isinstance(plan, dict) else {}
            faq_info = expansion.get("faq") or {}
            expected_count = int(faq_info.get("expected") or 0)
            created_count = int(faq_info.get("created") or 0)
            questions.append({
                "kind": kind,
                "technical_error": raw_messages[0] if raw_messages else "",
                "severity": "blocking",
                "human_summary": "O conjunto de FAQs esperado para este plano ainda nao esta completo.",
                "probable_cause": f"O plano espera {expected_count} FAQ(s) mas tem {created_count}.",
                "question": "Quer que a Sofia gere os FAQs faltantes a partir das copies/produtos terminais?",
                "options": [
                    {
                        "label": "Sim, gerar FAQs faltantes automaticamente",
                        "action": "regenerate_missing_faqs",
                        "payload": {},
                    },
                    {
                        "label": "Reduzir o numero esperado de FAQs",
                        "action": "lower_faq_target",
                        "payload": {"new_target": created_count},
                    },
                    {
                        "label": "Seguir sem completar os FAQs",
                        "action": "drop_faq_target",
                        "payload": {},
                    },
                ],
            })
            continue

        if kind == "offer_missing":
            product_nodes = [n for n in nodes if n.get("type") == "product"]
            options = []
            for p in product_nodes:
                options.append({
                    "label": f"Criar oferta abaixo de {_diagnostic_entry_label(p)}",
                    "action": "create_offer",
                    "payload": {"product_slug": p.get("slug")},
                })
            options.append({
                "label": "Seguir sem oferta neste plano",
                "action": "drop_offer_requirement",
                "payload": {},
            })
            questions.append({
                "kind": kind,
                "technical_error": raw_messages[0] if raw_messages else "",
                "severity": "blocking",
                "human_summary": "A sessao indicou preco/kit/variacao mas nenhuma oferta foi criada.",
                "probable_cause": "Quando o briefing menciona preco, kit ou plano, a Sofia espera ao menos uma oferta.",
                "question": "Quer criar uma oferta agora?",
                "options": options,
            })
            continue

        if kind == "rule_missing":
            questions.append({
                "kind": kind,
                "technical_error": raw_messages[0] if raw_messages else "",
                "severity": "blocking",
                "human_summary": "A sessao indicou regras comerciais (prazo, troca, pagamento) mas nenhuma regra foi criada.",
                "probable_cause": "Quando o briefing menciona regras comerciais, a Sofia espera ao menos uma regra acima.",
                "question": "Quer criar a regra comercial agora?",
                "options": [
                    {"label": "Criar regra abaixo da campanha", "action": "create_rule", "payload": {"scope": "campaign"}},
                    {"label": "Criar regra abaixo do briefing", "action": "create_rule", "payload": {"scope": "briefing"}},
                    {"label": "Seguir sem regra neste plano", "action": "drop_rule_requirement", "payload": {}},
                ],
            })
            continue

        if kind in {"offer_under_product", "audience_parent", "product_under_audience",
                    "product_invalid_parent", "copy_parent", "faq_parent", "rule_parent"}:
            for idx in affected_idxs:
                node = nodes_by_index.get(idx)
                if not node:
                    continue
                expected = node.get("expected_parent_types") or []
                label = _diagnostic_entry_label(node)
                ntype = node.get("type") or ""
                tech = next(
                    (m for m in raw_messages if isinstance(m, str) and f"entry[{idx}]" in m),
                    raw_messages[0] if raw_messages else "",
                )
                questions.append({
                    "kind": kind,
                    "technical_error": tech,
                    "affected_entry_index": idx,
                    "affected_title": label,
                    "affected_type": ntype,
                    "severity": "blocking",
                    "human_summary": f"\"{label}\" esta abaixo de um node que nao e permitido.",
                    "probable_cause": f"{_diagnostic_humanize_type(ntype)} so pode pertencer a {', '.join(expected) or 'um node estrutural'}.",
                    "question": f"Em qual node \"{label}\" deve ficar?",
                    "options": _sofia_options_for_parent_choice(node, nodes, existing, expected),
                })
            continue

        if kind == "cycle":
            for idx in affected_idxs:
                node = nodes_by_index.get(idx)
                if not node:
                    continue
                expected = node.get("expected_parent_types") or []
                label = _diagnostic_entry_label(node)
                tech = next(
                    (m for m in raw_messages if isinstance(m, str) and f"entry[{idx}]" in m),
                    raw_messages[0] if raw_messages else "",
                )
                questions.append({
                    "kind": kind,
                    "technical_error": tech,
                    "affected_entry_index": idx,
                    "affected_title": label,
                    "affected_type": node.get("type") or "",
                    "severity": "blocking",
                    "human_summary": f"\"{label}\" esta apontando para um pai que volta para ela mesma (ciclo).",
                    "probable_cause": "A cadeia metadata.parent_slug formou um loop.",
                    "question": f"Para qual node \"{label}\" deve apontar?",
                    "options": _sofia_options_for_parent_choice(node, nodes, existing, expected),
                })
            continue

        if kind == "missing_parent_slug":
            for idx in affected_idxs:
                node = nodes_by_index.get(idx)
                if not node:
                    continue
                expected = node.get("expected_parent_types") or []
                label = _diagnostic_entry_label(node)
                tech = next(
                    (m for m in raw_messages if isinstance(m, str) and f"entry[{idx}]" in m),
                    raw_messages[0] if raw_messages else "",
                )
                questions.append({
                    "kind": kind,
                    "technical_error": tech,
                    "affected_entry_index": idx,
                    "affected_title": label,
                    "affected_type": node.get("type") or "",
                    "severity": "blocking",
                    "human_summary": f"\"{label}\" foi criada sem indicar de onde ela vem.",
                    "probable_cause": "Falta declarar metadata.parent_slug ou aparecer em links[].",
                    "question": f"De qual node \"{label}\" depende?",
                    "options": _sofia_options_for_parent_choice(node, nodes, existing, expected),
                })
            continue

        if kind == "duplicate_slug":
            questions.append({
                "kind": kind,
                "technical_error": raw_messages[0] if raw_messages else "",
                "severity": "blocking",
                "human_summary": "Duas ou mais entradas estao usando o mesmo slug.",
                "probable_cause": "Slugs precisam ser unicos dentro do plano.",
                "question": "Quer que a Sofia gere slugs unicos automaticamente?",
                "options": [
                    {"label": "Sim, regerar slugs duplicados", "action": "regenerate_slugs", "payload": {}},
                    {"label": "Editar manualmente no plano", "action": "open_plan_editor", "payload": {}},
                ],
            })
            continue

        # Fallback: keep technical message visible but wrap in a generic question
        questions.append({
            "kind": kind,
            "technical_error": raw_messages[0] if raw_messages else "",
            "severity": "blocking",
            "human_summary": bucket.get("description") or "O validador identificou uma pendencia que precisa de decisao.",
            "probable_cause": bucket.get("description") or "",
            "question": "Como deseja resolver essa pendencia?",
            "options": [
                {"label": "Editar plano manualmente", "action": "open_plan_editor", "payload": {}},
                {"label": "Manter como pendencia", "action": "mark_pending", "payload": {}},
            ],
        })

    # Pos-processamento: garante que cada opcao saia com `prompt_to_sofia`
    # populado (o front usa esse prompt para enviar a mensagem certa para a
    # Sofia/tool). Caso o backend ainda nao expresse o prompt por opcao
    # diretamente, derivamos a partir de action+payload.
    for q in questions:
        for opt in q.get("options") or []:
            if opt.get("prompt_to_sofia"):
                continue
            prompt, ui_hook = _derive_sofia_option_prompt(q, opt)
            if prompt:
                opt["prompt_to_sofia"] = prompt
            if ui_hook and not opt.get("ui_hook"):
                opt["ui_hook"] = ui_hook

    return questions


def _derive_sofia_option_prompt(question: dict, option: dict) -> tuple[str, Optional[str]]:
    """Traduz action+payload de uma SofiaQuestionOption em um prompt que a
    Sofia (via tools quando habilitadas, ou via texto livre quando nao)
    entende como instrucao para mutar o plano.

    Devolve (prompt, ui_hook?). O front ja tem um fallback local equivalente;
    repetir aqui evita depender de versoes do front em sincronia.
    """
    payload = option.get("payload") or {}
    action = str(option.get("action") or "").strip().lower()
    target_slug = str(payload.get("slug") or "").strip()
    scope = str(payload.get("scope") or "").strip()
    product_slug = str(payload.get("product_slug") or "").strip()
    new_target = payload.get("new_target")
    affected_title = str(question.get("affected_title") or "").strip()
    affected_type = str(question.get("affected_type") or "").strip()
    if action == "upload_asset":
        return (
            "Vou subir um novo asset agora. Quando o upload terminar, conecte o asset a partir de "
            "session.asset_readings ao produto principal do plano (tool attach_session_asset) e "
            "marque como pendente_validacao.",
            "open_file_picker",
        )
    if action == "attach_existing_asset":
        return (
            f"Conecte o asset existente {target_slug or '(slug pendente)'} ao "
            f"{scope or 'produto principal'} do plano usando uses_asset (tool attach_session_asset "
            f"se vier de session.asset_readings, senao create_node content_type=asset).",
            None,
        )
    if action == "drop_asset_requirement":
        return (
            "Remova a exigencia de asset: chame set_expansion_policy(block='asset', "
            "count_per_parent=0, count_policy='per_parent').",
            None,
        )
    if action == "regenerate_missing_faqs":
        return (
            "Gere os FAQs faltantes a partir das copies/produtos terminais do plano atual usando "
            "create_node, mantendo o faq_parent_type configurado.",
            None,
        )
    if action == "lower_faq_target":
        new_t = new_target if isinstance(new_target, int) else 1
        return (
            f"Chame set_expansion_policy(block='faq', count_per_parent={new_t}). Mantenha os FAQs "
            f"ja gerados.",
            None,
        )
    if action == "drop_faq_target":
        return (
            "Remova a exigencia de gerar mais FAQs: set_expansion_policy(block='faq', "
            "count_policy='total'). Mantenha os FAQs ja criados.",
            None,
        )
    if action == "create_offer":
        return (
            f"Crie uma oferta abaixo do produto {product_slug or '(principal)'} usando create_node "
            f"com content_type='offer', parent_slug='{product_slug or '<produto principal>'}', "
            f"title/content concretos.",
            None,
        )
    if action == "drop_offer_requirement":
        return "Remova a exigencia de oferta deste plano e prossiga sem offers.", None
    if action == "create_rule":
        return (
            f"Crie uma regra comercial abaixo do {scope or 'briefing'} usando create_node com "
            f"content_type='rule' e parent_slug='{scope or 'briefing'}'.",
            None,
        )
    if action == "drop_rule_requirement":
        return "Remova a exigencia de rule deste plano e prossiga sem rules.", None
    if action == "change_parent":
        new_parent = str(payload.get("new_parent_slug") or "").strip()
        entry_slug = str(payload.get("entry_slug") or "").strip()
        if entry_slug and new_parent:
            return (
                f"Chame set_parent(slug='{entry_slug}', parent_slug='{new_parent}'). Re-valide "
                f"depois com validate_plan.",
                None,
            )
        return (
            f"Corrija o parent da entry {affected_title or affected_type} para alcancar a persona "
            f"(set_parent), depois validate_plan.",
            None,
        )
    if action == "regenerate_slugs":
        return (
            "Re-emita o knowledge_plan com slugs unicos: identifique slugs duplicados, sufixe com "
            "-2/-3/etc e refaca os links que apontavam para o slug antigo.",
            None,
        )
    if action == "open_plan_editor":
        return "", None
    if action == "mark_pending":
        return (
            "Mantenha o item como pendente_validacao; nao bloqueie o plano. Re-emita o "
            "knowledge_plan com o status atualizado.",
            None,
        )
    # Default: usa o label da opcao como hint para a Sofia.
    label = option.get("label") or ""
    if label:
        return (
            f"Aplique a opcao \"{label}\" para resolver \"{question.get('human_summary') or question.get('kind')}\". "
            "Use tools quando aplicavel e re-emita o knowledge_plan corrigido.",
            None,
        )
    return "", None


def _sofia_questions_markdown(sofia_questions: list[dict[str, Any]]) -> str:
    """Render Sofia questions as markdown so the modal can keep the legacy
    `questions_markdown` field useful when it is the only surface available."""
    if not sofia_questions:
        return ""
    lines = ["## O que falta decidir", ""]
    for idx, q in enumerate(sofia_questions, 1):
        lines.append(f"### {idx}. {q.get('human_summary') or q.get('question') or ''}")
        question_text = q.get("question") or ""
        if question_text:
            lines.append("")
            lines.append(question_text)
        options = q.get("options") or []
        if options:
            lines.append("")
            for opt in options:
                label = opt.get("label") or ""
                if label:
                    lines.append(f"- {label}")
        lines.append("")
    return "\n".join(lines).strip()


def build_plan_diagnostic(
    plan: dict,
    session: Optional[dict],
    violations: list[str],
) -> Optional[dict[str, Any]]:
    """Build a structured diagnostic for a blocked plan.

    Returns None when there are no blocking violations or no entries to classify.
    The shape is documented in dashboard/app/knowledge/capture/page.tsx (PlanState.diagnostic).
    """
    if not violations:
        return None
    if not isinstance(plan, dict):
        return None
    entries = plan.get("entries") or []
    if not isinstance(entries, list):
        return None

    slug_to_entry: dict[str, dict] = {}
    parent_by_child: dict[str, Optional[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if not slug:
            continue
        slug_to_entry[str(slug)] = entry
        parent_by_child[str(slug)] = _entry_parent_slug(entry)

    has_audience = any(_entry_type(e) == "audience" for e in entries if isinstance(e, dict))
    has_offer = any(_entry_type(e) == "offer" for e in entries if isinstance(e, dict))
    has_copy = any(_entry_type(e) == "copy" for e in entries if isinstance(e, dict))

    # Group violations first so we can attach raw messages to nodes.
    by_kind: dict[str, dict] = {}
    for violation in violations:
        kind, affected_idx = _diagnostic_classify_violation(violation)
        bucket = by_kind.setdefault(kind, {
            "kind": kind,
            "title": _DIAGNOSTIC_KIND_TITLES.get(kind, kind),
            "description": _DIAGNOSTIC_KIND_DESCRIPTIONS.get(kind, ""),
            "affected_entry_indexes": [],
            "raw_messages": [],
            "suggested_repair": _DIAGNOSTIC_KIND_REPAIRS.get(kind, ""),
        })
        if affected_idx is not None and affected_idx not in bucket["affected_entry_indexes"]:
            bucket["affected_entry_indexes"].append(affected_idx)
        bucket["raw_messages"].append(violation)

    issues_by_index: dict[int, list[str]] = {}
    for bucket in by_kind.values():
        for idx in bucket["affected_entry_indexes"]:
            issues_by_index.setdefault(idx, []).append(bucket["title"])

    nodes: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        ctype = _entry_type(entry)
        parent_slug = _entry_parent_slug(entry)
        parent_entry = slug_to_entry.get(parent_slug or "")
        parent_type = _entry_type(parent_entry) if parent_entry else None
        expected = _diagnostic_expected_parents(
            ctype,
            has_audience=has_audience,
            has_offer=has_offer,
            has_copy=has_copy,
        )

        status = "valid"
        node_issues: list[str] = list(issues_by_index.get(idx, []))
        slug_value = str(entry.get("slug") or "")
        is_top_level = ctype in SOFIA_TOP_LEVEL_TYPES

        reaches_persona = is_top_level
        has_cycle = False
        chain_path: list[str] = []
        if slug_value:
            reaches_persona, has_cycle, chain_path = _diagnostic_walk_parent_chain(slug_value, parent_by_child)
            if is_top_level:
                # top-level types do not need to walk further
                reaches_persona = True

        parent_is_persona_ref = parent_slug == "self"
        if has_cycle:
            status = "cycle"
        elif not is_top_level and not reaches_persona:
            status = "orphan"
        elif expected and parent_type and parent_type not in expected:
            status = "error"
        elif expected and not parent_entry and not is_top_level and not parent_is_persona_ref:
            status = "error"
            node_issues.append("Sem parent declarado")
        elif idx in issues_by_index:
            status = "warning"

        suggested_action: Optional[str] = None
        if status == "error" and expected:
            suggested_action = f"Escolher parent do tipo {expected[0]}"
        elif status == "cycle":
            suggested_action = "Quebrar o ciclo: aponte parent_slug para um node real acima"
        elif status == "orphan":
            suggested_action = "Reconectar entrada ate a persona"
        elif status == "warning":
            suggested_action = "Revisar pendencia relacionada"

        nodes.append({
            "entry_index": idx,
            "slug": slug_value,
            "title": _diagnostic_entry_title(entry),
            "type": ctype,
            "parent_slug": parent_slug,
            "parent_type": parent_type,
            "expected_parent_types": expected,
            "status": status,
            "issues": node_issues,
            "suggested_action": suggested_action,
            "chain_path": chain_path,
        })

    root_causes = list(by_kind.values())
    summary = {
        "entry_count": len(nodes),
        "blocked_count": len(violations),
        "cycle_count": sum(1 for n in nodes if n["status"] == "cycle"),
        "orphan_count": sum(1 for n in nodes if n["status"] == "orphan"),
        "error_count": sum(1 for n in nodes if n["status"] == "error"),
        "warning_count": sum(1 for n in nodes if n["status"] == "warning"),
        "valid_count": sum(1 for n in nodes if n["status"] == "valid"),
        "root_cause_count": len(root_causes),
    }
    sofia_questions = _sofia_questions_from_diagnostic(plan, session, nodes, by_kind)
    # Primary markdown surface = Sofia translation. The legacy generic
    # markdown is kept as a fallback when Sofia could not produce any
    # question (e.g. unknown violation kinds).
    questions_markdown = _sofia_questions_markdown(sofia_questions) or _diagnostic_questions_markdown(
        plan, nodes, root_causes, by_kind
    )

    return {
        "blocked": True,
        "summary": summary,
        "nodes": nodes,
        "root_causes": root_causes,
        "sofia_questions": sofia_questions,
        "questions_markdown": questions_markdown,
        "repair_suggestion": (
            "Arvore sugerida: persona -> brand -> briefing -> campaign -> audience -> "
            "product_group -> product -> copy -> faq"
        ),
    }


def _entry_type(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    return _normalize_content_type_alias(entry.get("content_type")) or ""


def _entry_metadata(entry: dict) -> dict:
    meta = entry.get("metadata")
    if isinstance(meta, dict):
        return meta
    meta = {}
    entry["metadata"] = meta
    return meta


def _entry_parent_slug(entry: dict) -> Optional[str]:
    parent = _entry_metadata(entry).get("parent_slug")
    return str(parent).strip() if parent else None


def _set_entry_parent_slug(entry: dict, parent_slug: Optional[str]) -> None:
    meta = _entry_metadata(entry)
    if parent_slug:
        meta["parent_slug"] = parent_slug
    else:
        meta.pop("parent_slug", None)


def _normalize_parent_slug_value(parent_slug: Optional[str], persona_slug: str) -> Optional[str]:
    raw = str(parent_slug or "").strip()
    if not raw:
        return None
    if raw.lower() in {"global", "root", "persona", "persona-root"}:
        return "self"
    if _slug_for_plan_entry(raw) == _slug_for_plan_entry(persona_slug or ""):
        return "self"
    return raw


_PARALLEL_BRANCH_RE = re.compile(
    r"\b(outputs?\s+paralelos?|galhos?\s+paralelos?|branches?\s+paralelos?|"
    r"separ(e|ar)\s+copy\s+e\s+faq|copys?\s+e\s+faqs?\s+como\s+galhos?|"
    r"faqs?\s+direto\s+no\s+produto|diretamente\s+abaixo\s+do\s+produto)\b",
    re.I,
)
_TECHNICAL_FAQ_RE = re.compile(
    r"\b(faq\s+t[eé]cnic[oa]|perguntas?\s+t[eé]cnicas?|d[uú]vidas?\s+t[eé]cnicas?|"
    r"factual\s+do\s+produto|sobre\s+especifica[cç][oõ]es|especifica[cç][oõ]es\s+do\s+produto)\b",
    re.I,
)


def _session_text_for_branch_policy(session: dict) -> str:
    parts = [str(session.get("context") or "")]
    for msg in session.get("messages") or []:
        if isinstance(msg, dict):
            parts.append(str(msg.get("content") or ""))
    return "\n".join(parts)


def _explicit_parallel_outputs_requested(session: dict) -> bool:
    return bool(_PARALLEL_BRANCH_RE.search(_session_text_for_branch_policy(session)))


def _technical_product_faq_requested(session: dict) -> bool:
    return bool(_TECHNICAL_FAQ_RE.search(_session_text_for_branch_policy(session)))


def _normalize_plan_parent_slugs(plan: dict, persona_slug: str) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    brand_slugs = {
        str(entry.get("slug") or "").strip()
        for entry in entries
        if _entry_type(entry) == "brand" and entry.get("slug")
    }
    for entry in plan.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        meta = _entry_metadata(entry)
        raw_parent = str(meta.get("parent_slug") or "").strip()
        if (
            raw_parent
            and _entry_type(entry) != "brand"
            and raw_parent in brand_slugs
            and _slug_for_plan_entry(raw_parent) == _slug_for_plan_entry(persona_slug or "")
        ):
            normalized_parent = raw_parent
        else:
            normalized_parent = _normalize_parent_slug_value(raw_parent, persona_slug)
        if normalized_parent:
            meta["parent_slug"] = normalized_parent
        else:
            meta.pop("parent_slug", None)


def _repair_canonical_parent_slugs(plan: dict) -> int:
    """Repair structural parents using the canonical tree order.

    The business graph is serial: campaign -> audience -> product_group ->
    product. This function repairs both missing parents and invalid shortcuts
    emitted by Sofia/LLM, such as product_group under campaign or product under
    audience while a product_group exists.
    """
    if not isinstance(plan, dict):
        return 0
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    if not entries:
        return 0

    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        ctype = _entry_type(entry)
        if ctype:
            by_type.setdefault(ctype, []).append(entry)

    canonical_parents: dict[str, tuple[str, ...]] = {
        "brand": ("persona",),
        "briefing": ("brand", "persona"),
        "campaign": ("briefing", "brand", "persona"),
        "audience": ("campaign", "briefing", "brand", "persona"),
        "product_group": ("audience",),
        "product": ("product_group",),
        "copy": ("product", "product_group", "campaign", "audience", "briefing", "brand"),
        "faq": ("copy", "product", "product_group", "campaign", "audience", "briefing", "brand"),
        "offer": ("product", "product_group", "campaign", "audience", "briefing", "brand"),
        "rule": ("campaign", "briefing", "brand", "persona"),
        "asset": ("product", "product_group", "copy", "faq", "campaign", "audience", "brand"),
    }

    repaired = 0
    slug_to_entry = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}

    def _current_parent_type(entry: dict) -> str:
        parent_slug = _entry_parent_slug(entry)
        if parent_slug == "self":
            return "persona"
        return _entry_type(slug_to_entry.get(parent_slug or "") or {})

    def _parent_map_with(entry_slug: str, parent_slug: str) -> dict[str, str]:
        parent_by_child = {
            str(candidate.get("slug")): str(_entry_parent_slug(candidate) or "")
            for candidate in entries
            if candidate.get("slug") and _entry_parent_slug(candidate)
        }
        parent_by_child[entry_slug] = parent_slug
        return parent_by_child

    def _would_create_cycle(entry: dict, parent_candidate: dict) -> bool:
        entry_slug = str(entry.get("slug") or "").strip()
        parent_slug = str(parent_candidate.get("slug") or "").strip()
        if not entry_slug or not parent_slug or entry_slug == parent_slug:
            return True
        parent_by_child = _parent_map_with(entry_slug, parent_slug)
        seen: set[str] = set()
        cursor = entry_slug
        while cursor:
            if cursor in {"self", "persona", "root", "global"}:
                return False
            if cursor in seen:
                return True
            seen.add(cursor)
            cursor = parent_by_child.get(cursor, "")
        return False

    for _ in range(3):
        changed_this_pass = 0
        for entry in entries:
            ctype = _entry_type(entry)
            if not ctype or ctype == "persona":
                continue
            allowed_parent_types = canonical_parents.get(ctype, ())
            if _entry_parent_slug(entry) and (
                ctype == "brand"
                or not allowed_parent_types
                or _current_parent_type(entry) in allowed_parent_types
            ):
                continue
            metadata = _entry_metadata(entry)
            parent_candidate: Optional[dict] = None
            for parent_type in canonical_parents.get(ctype, ()):
                candidates = [candidate for candidate in by_type.get(parent_type, []) if candidate is not entry and candidate.get("slug")]
                if not candidates:
                    continue
                ordered: list[dict] = []
                picked = _best_parent_by_slug(entry, candidates)
                if picked is not None:
                    ordered.append(picked)
                ordered.extend(candidate for candidate in reversed(candidates) if candidate is not picked)
                for candidate in ordered:
                    if not _would_create_cycle(entry, candidate):
                        parent_candidate = candidate
                        break
                if parent_candidate and parent_candidate.get("slug"):
                    break
            if parent_candidate and parent_candidate.get("slug"):
                old_parent = metadata.get("parent_slug")
                metadata["parent_slug"] = str(parent_candidate.get("slug"))
                metadata["parent_inferred"] = True
                metadata["parent_inferred_from"] = parent_candidate.get("content_type") or "unknown"
                if old_parent != metadata["parent_slug"]:
                    changed_this_pass += 1
                continue
            if ctype == "brand":
                old_parent = metadata.get("parent_slug")
                metadata["parent_slug"] = "self"
                metadata["parent_inferred"] = True
                metadata["parent_inferred_from"] = "persona"
                if old_parent != "self":
                    changed_this_pass += 1
        repaired += changed_this_pass
        if not changed_this_pass:
            break
    return repaired


def _knowledge_plan_title_from_session(session: dict) -> str:
    cls = session.get("classification") or {}
    if cls.get("title"):
        return str(cls["title"])
    context = str(session.get("context") or "")
    source_url = _source_url_from_context(context)
    if source_url:
        return f"Captura de {source_url.split('//')[-1]}"
    return "Plano de conhecimento"


def _requested_variation_count(session: dict, block_id: str, default: int) -> int:
    counts = session.get("initial_block_counts")
    if isinstance(counts, dict) and block_id in counts:
        try:
            return max(int(counts.get(block_id) or 0), 0)
        except Exception:
            pass
    context = str(session.get("context") or "")
    pattern = rf"^\s*-\s*{re.escape(block_id)}:\s*(\d+)\s+vari"
    match = re.search(pattern, context, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        try:
            return max(int(match.group(1)), 0)
        except Exception:
            pass
    return default


def _has_explicit_variation_count(session: dict, block_id: str) -> bool:
    counts = session.get("initial_block_counts")
    if isinstance(counts, dict) and block_id in counts:
        return True
    context = str(session.get("context") or "")
    pattern = rf"^\s*-\s*{re.escape(block_id)}:\s*(\d+)\s+vari"
    return bool(re.search(pattern, context, flags=re.IGNORECASE | re.MULTILINE))


def _normalize_plan_entry(entry: dict) -> dict:
    normalized = dict(entry or {})
    normalized["content_type"] = _normalize_content_type_alias(_entry_type(normalized)) or "other"
    normalized["title"] = str(normalized.get("title") or "").strip() or "Conhecimento"
    normalized["slug"] = _slug_for_plan_entry(str(normalized.get("slug") or normalized["title"]))
    normalized["status"] = str(normalized.get("status") or "pendente_validacao").strip() or "pendente_validacao"
    content = str(normalized.get("content") or "").strip()
    normalized["content"] = content or normalized["title"]
    tags = normalized.get("tags") or []
    normalized["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
    metadata = dict(normalized.get("metadata") or {})
    ctype = normalized["content_type"]
    if ctype == "briefing":
        metadata.setdefault("briefing_scope", "global")
        metadata.setdefault("governs_children", True)
    elif ctype == "audience":
        metadata.setdefault("audience_source", "manual")
        metadata.setdefault("import_page_url", None)
        metadata.setdefault("leads_page_url", None)
        metadata.setdefault("lead_segment_id", None)
        metadata.setdefault("crm_filters", {})
    elif ctype == "product":
        metadata.setdefault("product_source", "manual")
        metadata.setdefault("product_external_id", None)
        metadata.setdefault("product_page_url", None)
        metadata.setdefault("price_status", "pending_validation")
        metadata.setdefault("stock_status", "unknown")
    elif ctype == "offer":
        metadata.setdefault("offer_source", "manual")
        metadata.setdefault("price_status", "pending_validation")
        metadata.setdefault("stock_status", "unknown")
    elif ctype in {"entity", "other"} and str(metadata.get("entity_role") or "").strip():
        metadata.setdefault("entity_structural", metadata.get("entity_role") in {"product_group", "audience_group", "category_group"})
    normalized["metadata"] = metadata
    return normalized


def _relation_type_for_parent(parent_type: str, child_type: str) -> str:
    mapping = {
        ("persona", "brand"): "contains",
        ("persona", "briefing"): "contains",
        ("persona", "campaign"): "contains",
        ("persona", "audience"): "contains",
        ("persona", "product"): "contains",
        ("brand", "briefing"): "contains",
        ("brand", "product"): "contains",
        ("brand", "audience"): "contains",
        ("brand", "campaign"): "contains",
        ("briefing", "campaign"): "briefed_by",
        ("campaign", "audience"): "targets_audience",
        ("briefing", "audience"): "contains",
        ("briefing", "product"): "contains",
        ("product", "briefing"): "contains",
        ("audience", "briefing"): "contains",
        ("audience", "product_group"): "contains",
        ("campaign", "product_group"): "contains",
        ("briefing", "product_group"): "contains",
        ("brand", "product_group"): "contains",
        ("product_group", "product"): "contains",
        ("product_group", "copy"): "supports_copy",
        ("product_group", "faq"): "answers_question",
        ("product_group", "asset"): "uses_asset",
        ("audience", "product"): "offers_product",
        ("product", "offer"): "contains",
        ("offer", "copy"): "supports_copy",
        ("offer", "faq"): "answers_question",
        ("campaign", "rule"): "contains",
        ("briefing", "rule"): "contains",
        ("brand", "rule"): "contains",
        ("rule", "faq"): "answers_question",
        ("entity", "product"): "contains",
        ("other", "product"): "contains",
        ("brand", "entity"): "contains",
        ("brand", "other"): "contains",
        ("briefing", "entity"): "contains",
        ("briefing", "other"): "contains",
        ("audience", "product"): "offers_product",
        ("product", "faq"): "answers_question",
        ("product", "copy"): "supports_copy",
        ("copy", "faq"): "answers_question",
        ("product", "asset"): "uses_asset",
    }
    return mapping.get((parent_type, child_type), "contains")


def _is_b2b_audience(entry: dict) -> bool:
    blob = " ".join([
        str(entry.get("title") or ""),
        str(entry.get("content") or ""),
        " ".join(entry.get("tags") or []),
    ]).lower()
    return any(token in blob for token in ("varej", "revend", "lojist", "atacad", "empreended"))


def _is_b2b_faq(entry: dict) -> bool:
    blob = " ".join([
        str(entry.get("title") or ""),
        str(entry.get("content") or ""),
        " ".join(entry.get("tags") or []),
    ]).lower()
    return any(token in blob for token in ("quantidade minima", "pedido minimo", "revend", "varej", "atacad"))


def _known_colors_from_session(session: dict, plan: dict) -> list[str]:
    blob_parts = [str(session.get("context") or ""), json.dumps(plan, ensure_ascii=False)]
    blob = " ".join(blob_parts).lower()
    known = []
    for color in ("preta", "preto", "vermelha", "vermelho", "roxa", "roxo", "azul", "rosa", "branca", "branco"):
        if color in blob:
            normalized = color[:-1] + "a" if color.endswith("o") else color
            if normalized not in known:
                known.append(normalized)
    return known


def _plan_blob(session: dict, plan: Optional[dict] = None) -> str:
    parts = [str(session.get("context") or "")]
    for message in session.get("messages") or []:
        if isinstance(message, dict):
            parts.append(str(message.get("content") or ""))
    if plan:
        try:
            parts.append(json.dumps(plan, ensure_ascii=False))
        except Exception:
            pass
    return "\n".join(parts)


def _commercial_offer_specs(session: dict, plan: dict) -> list[dict[str, Any]]:
    blob = _plan_blob(session, plan)
    specs: dict[int, dict[str, Any]] = {}
    for match in re.finditer(
        r"(?P<qty>\d+)\s*(?:pe[cç]as?|unidades?|itens?)\s*(?:por|=|:|-)?\s*R\$\s*(?P<price>\d{1,4}(?:[.,]\d{2})?)",
        blob,
        flags=re.IGNORECASE,
    ):
        qty = int(match.group("qty"))
        price = match.group("price")
        if qty <= 0:
            continue
        specs[qty] = {"quantity": qty, "price": price, "audience_role": None}
    for match in re.finditer(
        r"(?P<qty>\d+)\s*(?:pe[cç]as?|unidades?|itens?)",
        blob,
        flags=re.IGNORECASE,
    ):
        qty = int(match.group("qty"))
        if qty > 0:
            specs.setdefault(qty, {"quantity": qty, "price": None, "audience_role": None})
    lowered = blob.lower()
    for qty, spec in specs.items():
        spec.setdefault("label", f"{qty} unidade" if qty == 1 else f"{qty} unidades")
        if not spec.get("audience_role"):
            if qty == 1 and re.search(r"\b(?:1\s*(?:pe[cç]a|unidade).*?(?:cliente|publico|p[uú]blico)\s+final|(?:cliente|publico|p[uú]blico)\s+final.*?1\s*(?:pe[cç]a|unidade))\b", lowered, re.I):
                spec["audience_role"] = "final"
            elif qty >= 5 and re.search(r"\b(?:5|10)\s*(?:pe[cç]as?|unidades?)?.*?(?:empreended|revend|atacad|lojist)|(?:empreended|revend|atacad|lojist).*?(?:5|10)\s*(?:pe[cç]as?|unidades?)?\b", lowered, re.I):
                spec["audience_role"] = "b2b"
            else:
                spec["audience_role"] = "any"
    return [specs[key] for key in sorted(specs)]


def _explicit_copy_per_offer_requested(session: dict) -> bool:
    blob = _plan_blob(session).lower()
    return bool(re.search(
        r"\b(?:copy|copies|copys?)\s+(?:diferentes?\s+)?(?:para\s+)?cada\s+oferta\b|"
        r"\buma\s+copy\s+por\s+oferta\b|"
        r"\bcopy_policy\s*[:=]\s*per_offer\b",
        blob,
        re.I,
    ))


def _offers_required(session: dict, plan: dict) -> bool:
    if _requested_variation_count(session, "offer", 0) > 0:
        return True
    specs = _commercial_offer_specs(session, plan)
    if specs:
        return True
    blob = _plan_blob(session, plan).lower()
    return bool(re.search(r"\b(pre[cç]o\s+diferente|quantidade\s+diferente|pacote|plano\s+(?:comercial|de\s+assinatura|de\s+compra)|assinatura|bundle|combo|op[cç][aã]o\s+de\s+compra|vers[aã]o\s+de\s+compra|condi[cç][aã]o\s+comercial\s+(?:especifica|diferente|propria)|varia[cç][aã]o\s+comercial|categoria\s+de\s+oferta)\b", blob))


def _rule_required(session: dict, plan: dict) -> bool:
    if _requested_variation_count(session, "rule", 0) > 0:
        return True
    blob = _plan_blob(session, plan).lower()
    return bool(re.search(r"\b(regra\s+comercial\s+obrigatoria|politica\s+comercial\s+obrigatoria|n[aã]o inventar|n[aã]o prometer)\b", blob))


def _is_final_audience(entry: dict) -> bool:
    blob = " ".join([
        str(entry.get("title") or ""),
        str(entry.get("content") or ""),
        " ".join(entry.get("tags") or []),
    ]).lower()
    return any(token in blob for token in ("cliente final", "clientes finais", "consumidor", "final"))


def _offer_applies_to_audience(spec: dict[str, Any], audience: Optional[dict]) -> bool:
    role = spec.get("audience_role") or "any"
    if role == "any" or not audience:
        return True
    if role == "b2b":
        return _is_b2b_audience(audience)
    if role == "final":
        return _is_final_audience(audience) or not _is_b2b_audience(audience)
    return True


def _dedupe_slug(base: str, used: set[str]) -> str:
    raw = _slug_for_plan_entry(base)
    if raw not in used:
        used.add(raw)
        return raw
    idx = 2
    while f"{raw}-{idx}" in used:
        idx += 1
    slug = f"{raw}-{idx}"
    used.add(slug)
    return slug


def _governing_scope_slug(entries: list[dict]) -> Optional[str]:
    for ctype in ("campaign", "briefing", "brand"):
        found = next((entry for entry in entries if _entry_type(entry) == ctype and entry.get("slug")), None)
        if found:
            return str(found.get("slug"))
    return "self"


def _ensure_governing_rule(plan: dict, session: dict) -> None:
    if not _rule_required(session, plan):
        return
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    parent_slug = _governing_scope_slug(entries)
    rules = [entry for entry in entries if _entry_type(entry) == "rule"]
    if rules:
        for rule in rules:
            if not _entry_parent_slug(rule):
                _set_entry_parent_slug(rule, parent_slug)
            meta = _entry_metadata(rule)
            meta.setdefault("governs_children", True)
            meta.setdefault("rule_scope", "campaign")
            meta.setdefault("structural_before_faq", True)
        return
    rule = _normalize_plan_entry({
        "content_type": "rule",
        "title": "Regra comercial pendente de validacao",
        "slug": "rule-regra-comercial-pendente",
        "status": "pendente_validacao",
        "content": "Consolidar a regra comercial confirmada pelo operador ou pela fonte. Nao inventar preco, estoque, prazo, disponibilidade, lote minimo ou condicao comercial sem validacao.",
        "tags": ["rule", "comercial", "pending-validation"],
        "metadata": {"parent_slug": parent_slug, "governs_children": True, "rule_scope": "campaign", "structural_before_faq": True},
    })
    entries.append(rule)
    plan["entries"] = entries


def _ensure_offer_entries(plan: dict, session: dict) -> None:
    if not _offers_required(session, plan):
        return
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
    requested_offer_count = max(0, _requested_variation_count(session, "offer", 0))
    specs = _commercial_offer_specs(session, plan)
    if not specs:
        specs = [{"label": f"opcao comercial {idx}", "quantity": None, "price": None, "audience_role": "any"} for idx in range(1, max(1, requested_offer_count) + 1)]
    elif requested_offer_count > len(specs):
        for idx in range(len(specs) + 1, requested_offer_count + 1):
            specs.append({"label": f"opcao comercial {idx}", "quantity": None, "price": None, "audience_role": "any"})
    used = {str(entry.get("slug")) for entry in entries if entry.get("slug")}
    existing_offer_keys = {
        (_entry_parent_slug(entry), str(_entry_metadata(entry).get("offer_key") or _entry_metadata(entry).get("quantity") or "generic"))
        for entry in entries
        if _entry_type(entry) == "offer"
    }
    new_entries: list[dict] = []
    for product in [entry for entry in entries if _entry_type(entry) == "product" and entry.get("slug")]:
        audience = by_slug.get(_entry_parent_slug(product) or "")
        for spec in specs:
            if not _offer_applies_to_audience(spec, audience):
                continue
            qty = spec.get("quantity")
            offer_key = str(qty if qty is not None else spec.get("label") or "generic")
            key = (str(product.get("slug")), offer_key)
            if key in existing_offer_keys:
                continue
            label = str(spec.get("label") or "opcao comercial")
            title = f"{product.get('title')} - {label}"
            if spec.get("price"):
                title = f"{title} R$ {spec['price']}"
            offer = _normalize_plan_entry({
                "content_type": "offer",
                "title": title,
                "slug": _dedupe_slug(f"offer-{product.get('slug')}-{label}", used),
                "status": "pendente_validacao",
                "content": f"Oferta comercial {label} para {product.get('title')}. Preco informado: {spec.get('price') or 'pendente de validacao'}.",
                "tags": ["offer", "commercial", "pending-validation"],
                "metadata": {
                    "parent_slug": str(product.get("slug")),
                    "quantity": qty,
                    "price": spec.get("price"),
                    "offer_key": offer_key,
                    "audience_role": spec.get("audience_role"),
                    "commercial_offer": True,
                },
            })
            new_entries.append(offer)
    if new_entries:
        entries.extend(new_entries)
        plan["entries"] = entries


def _ensure_copies_for_offers(plan: dict, session: Optional[dict] = None) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    offers = [entry for entry in entries if _entry_type(entry) == "offer" and entry.get("slug")]
    if not offers:
        return
    if _explicit_copy_per_offer_requested(session or {}):
        _ensure_copies_per_explicit_offer(plan)
        return
    used = {str(entry.get("slug")) for entry in entries if entry.get("slug")}
    by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
    copies = [entry for entry in entries if _entry_type(entry) == "copy"]
    offers_by_product: dict[str, list[dict]] = {}
    for offer in offers:
        product_slug = _entry_parent_slug(offer) or ""
        if product_slug:
            offers_by_product.setdefault(product_slug, []).append(offer)

    copies_by_product: dict[str, list[dict]] = {}
    offer_copy_templates: dict[str, list[dict]] = {}
    for copy in copies:
        parent_slug = _entry_parent_slug(copy)
        parent = by_slug.get(parent_slug or "")
        if _entry_type(parent) == "product":
            copies_by_product.setdefault(parent_slug or "", []).append(copy)
        elif _entry_type(parent) == "offer":
            product_slug = _entry_parent_slug(parent) or ""
            if product_slug:
                offer_copy_templates.setdefault(product_slug, []).append(copy)

    remove_copy_slugs: set[str] = set()
    for product_slug, product_offers in offers_by_product.items():
        existing = copies_by_product.get(product_slug) or []
        templates = existing or offer_copy_templates.get(product_slug) or copies
        template = templates[0] if templates else {}
        product = by_slug.get(product_slug) or {}
        grouped_offer_slugs = [str(offer.get("slug")) for offer in product_offers if offer.get("slug")]
        if existing:
            keep = existing[0]
            meta = _entry_metadata(keep)
            meta["parent_slug"] = product_slug
            meta["copy_policy"] = "per_product_context"
            meta["grouped_offer_slugs"] = grouped_offer_slugs
            meta["grouped_offer_count"] = len(grouped_offer_slugs)
            for extra in existing[1:]:
                remove_copy_slugs.add(str(extra.get("slug") or ""))
        else:
            title = str((template or {}).get("title") or f"Copy para {product.get('title') or product_slug}")
            copy = _normalize_plan_entry({
                **(template if isinstance(template, dict) else {}),
                "content_type": "copy",
                "title": title,
                "slug": _dedupe_slug(f"copy-{product_slug}", used),
                "status": "pendente_validacao",
                "content": str((template or {}).get("content") or f"Mensagem comercial para {product.get('title') or product_slug}, considerando as ofertas agrupadas do contexto."),
                "tags": list(dict.fromkeys([*((template or {}).get("tags") or []), "copy", "product-context"])),
                "metadata": {
                    **((template or {}).get("metadata") or {}),
                    "parent_slug": product_slug,
                    "copy_policy": "per_product_context",
                    "grouped_offer_slugs": grouped_offer_slugs,
                    "grouped_offer_count": len(grouped_offer_slugs),
                },
            })
            entries.append(copy)
            copies.append(copy)
        for offer_copy in offer_copy_templates.get(product_slug, []):
            remove_copy_slugs.add(str(offer_copy.get("slug") or ""))
    if remove_copy_slugs:
        entries[:] = [
            entry for entry in entries
            if _entry_type(entry) != "copy" or str(entry.get("slug") or "") not in remove_copy_slugs
        ]
    plan["entries"] = entries


def _ensure_copies_per_explicit_offer(plan: dict) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    offers = [entry for entry in entries if _entry_type(entry) == "offer" and entry.get("slug")]
    used = {str(entry.get("slug")) for entry in entries if entry.get("slug")}
    copies = [entry for entry in entries if _entry_type(entry) == "copy"]
    for offer in offers:
        offer_slug = str(offer.get("slug"))
        if any(_entry_parent_slug(copy) == offer_slug for copy in copies):
            continue
        template = copies[0] if copies else {}
        title = str((template or {}).get("title") or f"Copy para {offer.get('title')}")
        copy = _normalize_plan_entry({
            **(template if isinstance(template, dict) else {}),
            "content_type": "copy",
            "title": title if str(offer.get("title") or "") in title else f"{title} - {offer.get('title')}",
            "slug": _dedupe_slug(f"copy-{offer_slug}", used),
            "status": "pendente_validacao",
            "content": str((template or {}).get("content") or f"Mensagem comercial para {offer.get('title')}."),
            "tags": list(dict.fromkeys([*((template or {}).get("tags") or []), "copy", "offer"])),
            "metadata": {**((template or {}).get("metadata") or {}), "parent_slug": offer_slug, "copy_policy": "per_offer", "copied_for_offer": True},
        })
        entries.append(copy)
        copies.append(copy)
    plan["entries"] = entries


def _reparent_copies_to_offers(plan: dict, session: Optional[dict] = None) -> None:
    if not _explicit_copy_per_offer_requested(session or {}):
        return
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    offers = [entry for entry in entries if _entry_type(entry) == "offer" and entry.get("slug")]
    if not offers:
        return
    by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
    for copy in [entry for entry in entries if _entry_type(entry) == "copy"]:
        parent = by_slug.get(_entry_parent_slug(copy) or "")
        if _entry_type(parent or {}) == "offer":
            continue
        picked = _best_parent_by_slug(copy, offers) or offers[0]
        if picked and picked.get("slug"):
            _set_entry_parent_slug(copy, str(picked.get("slug")))
    plan["entries"] = entries


def _faq_leaf_entries(plan: dict) -> list[dict]:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    return [
        entry for entry in entries
        if _entry_type(entry) == "copy" and entry.get("slug")
    ] or [
        entry for entry in entries
        if _entry_type(entry) == "offer" and entry.get("slug")
    ] or [
        entry for entry in entries
        if _entry_type(entry) == "product" and entry.get("slug")
    ]


def _branch_chain_to_parent(parent: dict, entries_by_slug: dict[str, dict]) -> list[dict]:
    chain: list[dict] = []
    current = parent
    seen: set[str] = set()
    while current and current.get("slug") and str(current.get("slug")) not in seen and len(chain) < 16:
        current_slug = str(current.get("slug"))
        seen.add(current_slug)
        chain.append(current)
        current = entries_by_slug.get(_entry_parent_slug(current) or "")
    return list(reversed(chain))


def _entry_context_line(label: str, entry: Optional[dict]) -> str:
    if not entry:
        return f"- {label}: nao informado"
    title = str(entry.get("title") or "").strip() or str(entry.get("slug") or "").strip()
    content = re.sub(r"\s+", " ", str(entry.get("content") or "")).strip()
    if len(content) > 220:
        content = content[:217].rstrip() + "..."
    return f"- {label}: {title}" + (f" - {content}" if content else "")


def _branch_entries_by_type(branch_chain: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entry in branch_chain:
        ctype = _entry_type(entry)
        if ctype and ctype not in out:
            out[ctype] = entry
    return out


def _branch_semantic_tags(branch_chain: list[dict]) -> list[str]:
    tags: list[str] = []
    for entry in branch_chain:
        tags.extend(str(tag).strip() for tag in (entry.get("tags") or []) if str(tag).strip())
        ctype = _entry_type(entry)
        if ctype:
            tags.append(ctype)
    return list(dict.fromkeys(tags))[:24]


def build_faq_markdown_from_branch(payload: dict[str, Any]) -> dict[str, Any]:
    branch_path = payload.get("source_branch_path") or []
    entries = [item for item in branch_path if isinstance(item, dict)]
    by_type: dict[str, list[dict]] = {}
    for item in entries:
        ctype = str(item.get("content_type") or item.get("node_type") or "")
        if ctype:
            by_type.setdefault(ctype, []).append(item)
    terminal = branch_path[-1] if branch_path else {}
    terminal_title = str(terminal.get("title") or terminal.get("slug") or "atendimento").strip()
    question_count = max(8, int(payload.get("question_count") or 8))
    tags = [str(tag).strip() for tag in (payload.get("semantic_tags") or []) if str(tag).strip()]

    def text_of(entry: Optional[dict], field: str, fallback: str = "") -> str:
        return re.sub(r"\s+", " ", str((entry or {}).get(field) or fallback)).strip()

    def compact(value: str, fallback: str = "informacao pendente de validacao") -> str:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not value:
            return fallback
        value = re.sub(r"\b(arvore|árvore|grafo|galho|node|branch|regra|estrutura|conhecimento conectado)\b", "informacao", value, flags=re.I)
        return value if len(value) <= 260 else value[:257].rstrip() + "..."

    def commercial_facts(*items: Optional[dict], pattern: str) -> list[str]:
        blob = " ".join(
            str((entry or {}).get("content") or "") + " " + str((entry or {}).get("title") or "")
            for entry in items
            if entry
        )
        return list(dict.fromkeys(match.group(0) for match in re.finditer(pattern, blob, flags=re.IGNORECASE)))

    audiences = by_type.get("audience") or []
    products = by_type.get("product") or ([terminal] if terminal else [])
    offers = by_type.get("offer") or []
    copies = by_type.get("copy") or []
    briefings = by_type.get("briefing") or []
    campaigns = by_type.get("campaign") or []
    rules = [compact(str(rule), "") for rule in (payload.get("rules") or []) if str(rule).strip()]

    audience_titles = [text_of(item, "title") for item in audiences if text_of(item, "title")]
    product_titles = [text_of(item, "title") for item in products if text_of(item, "title")]
    offer_titles = [text_of(item, "title") for item in offers if text_of(item, "title")]
    copy_context = compact(" ".join(text_of(item, "content") for item in copies), "")
    briefing_context = compact(" ".join(text_of(item, "content") for item in [*briefings, *campaigns]), "")
    prices = commercial_facts(*products, *offers, *briefings, *campaigns, pattern=r"R\$\s*\d{1,6}(?:[.,]\d{2})?")
    quantities = commercial_facts(*offers, *products, *briefings, *campaigns, pattern=r"\b\d+\s*(?:pe[cç]as?|unidades?)\b")
    has_b2b = any(_is_b2b_audience(item) for item in audiences) or any(re.search(r"revend|empreend|atacad|lojist", text_of(item, "content") + " " + text_of(item, "title"), re.I) for item in audiences)
    has_final = any(_is_final_audience(item) for item in audiences)

    product_label = ", ".join(product_titles[:4]) or "os produtos"
    audience_label = ", ".join(audience_titles[:4]) or "cada perfil de cliente"
    offer_label = "; ".join(offer_titles[:6]) or ("; ".join(quantities[:6]) if quantities else "opcoes comerciais confirmadas pelo atendimento")
    price_label = ", ".join(prices[:6]) if prices else "valores a confirmar com a equipe"
    commercial_limits = "; ".join(rules[:2]) or "a equipe confirma valores, estoque, disponibilidade e condicoes antes de finalizar"

    question_templates: list[tuple[str, str]] = [
        ("catalog", "Quais produtos estao disponiveis?"),
        ("audience", "Esse kit e indicado para quem quer revender?"),
        ("final", "Posso comprar apenas uma peca para uso proprio?"),
        ("quantity", "Quais opcoes de quantidade eu posso pedir?"),
        ("price", "Quais valores posso considerar?"),
        ("stock", "Tem estoque e prazo de envio confirmados?"),
        ("benefit", "Por que esse produto pode ser uma boa opcao?"),
        ("order", "Como faco para fechar o pedido?"),
        ("compare", "Qual kit combina melhor comigo?"),
        ("payment", "As condicoes de pagamento ja estao definidas?"),
        ("support", "Posso tirar duvidas pelo WhatsApp antes de comprar?"),
        ("limits", "O que precisa ser confirmado antes de finalizar?"),
    ]

    def answer_for(kind: str) -> str:
        if kind == "catalog":
            return f"No momento, o atendimento trabalha com {product_label}. A equipe confirma disponibilidade, cores, tamanhos e valores antes de fechar o pedido."
        if kind == "audience":
            if has_b2b:
                return f"Sim. Esse kit pode ser uma boa opcao para quem quer revender, porque permite comprar em maior quantidade e montar uma oferta com apelo comercial. Antes de fechar, a equipe confirma {price_label} e as condicoes atuais."
            return "Pode ser indicado para revenda se houver quantidade e condicoes comerciais confirmadas pela equipe."
        if kind == "final":
            if has_final or any("1" in item for item in quantities):
                return "Sim. Quando houver opcao de 1 peca, ela e indicada para cliente final ou para quem quer experimentar antes de comprar mais unidades."
            return "A compra de unidade avulsa precisa ser confirmada com o atendimento, junto com valor e disponibilidade."
        if kind == "quantity":
            return f"As opcoes registradas sao: {offer_label}. O atendimento confirma qual opcao esta disponivel para {audience_label}."
        if kind == "benefit":
            return f"O principal e apresentar {product_label} de forma simples, destacando moda feminina, tecido modal, fabricacao propria e preco acessivel quando esses pontos estiverem confirmados."
        if kind == "price":
            return f"Os valores citados sao {price_label}. Como valores podem variar por quantidade, disponibilidade e condicao comercial, a equipe confirma antes de enviar a resposta final."
        if kind == "stock":
            return "Estoque, prazo e disponibilidade precisam ser verificados no momento do atendimento. A resposta segura e dizer que a equipe vai confirmar antes de concluir o pedido."
        if kind == "order":
            return "A cliente pode informar o kit desejado, quantidade e perfil da compra. Depois disso, a equipe confirma valor, disponibilidade, forma de pagamento e proximo passo."
        if kind == "compare":
            return f"A melhor opcao depende do objetivo da compra. Para uso proprio, uma unidade costuma ser suficiente; para revenda, kits maiores podem fazer mais sentido quando disponiveis."
        if kind == "payment":
            return "As condicoes de pagamento devem ser confirmadas pela equipe antes de responder, principalmente quando houver kit, desconto por quantidade ou pedido para revenda."
        if kind == "support":
            return "Sim. O WhatsApp pode tirar duvidas sobre produtos, quantidades, valores e disponibilidade antes da cliente decidir."
        if kind == "limits":
            return f"Antes de finalizar, confirme: {commercial_limits}. Evite prometer preco fechado, estoque, prazo, desconto ou disponibilidade sem checagem."
        return f"Use uma resposta curta e natural. {copy_context or briefing_context or 'Confirme os detalhes comerciais antes de orientar a compra.'}"

    lines = [
        f"# FAQ de atendimento - {terminal_title}",
        "",
        "## Referencia de atendimento",
        f"- Publico: {audience_label}",
        f"- Produtos: {product_label}",
        f"- Opcoes comerciais: {offer_label}",
        f"- Valores: {price_label}",
        f"- Confirmar antes de fechar: {commercial_limits}",
        "",
        "## Tags",
    ]
    lines.extend([f"- {tag}" for tag in tags] or ["- faq", "- golden-dataset"])
    lines.extend(["", "## Perguntas e respostas"])
    for idx in range(1, question_count + 1):
        if idx <= len(question_templates):
            kind, question = question_templates[idx - 1]
        else:
            kind = "limits"
            question = f"O que mais devo confirmar antes de responder a cliente {idx}?"
        lines.extend(["", f"### {idx}. {question}", "", f"**Resposta:** {answer_for(kind)}"])
    body = "\n".join(lines).strip()
    branch_slug = str(terminal.get("slug") or _slug_for_plan_entry(terminal_title))
    body = re.sub(r"\b(arvore|árvore|grafo|galho|node|branch|regra|estrutura|conhecimento conectado)\b", "informacao", body, flags=re.I)
    return {
        "content_type": "faq",
        "title": f"FAQ de atendimento - {terminal_title}",
        "slug": f"faq-golden-dataset-{branch_slug}",
        "body_markdown": body,
        "question_count": question_count,
        "source_branch_path": branch_path,
        "status": "pendente_validacao",
    }


def _build_faq_golden_dataset_entry(parent: dict, entries_by_slug: dict[str, dict], used: set[str], session: dict) -> dict:
    branch_chain = _branch_chain_to_parent(parent, entries_by_slug)
    by_type = _branch_entries_by_type(branch_chain)
    persona_slug = str((session.get("persona_slug") or (session.get("classification") or {}).get("persona_slug") or "persona")).strip()
    persona = {
        "content_type": "persona",
        "slug": persona_slug,
        "title": persona_slug,
        "content": f"Persona {persona_slug}",
    }
    source_branch_path = [
        {"content_type": "persona", "slug": persona_slug, "title": persona_slug, "content": f"Persona {persona_slug}"},
        *[
            {
                "content_type": _entry_type(entry),
                "slug": entry.get("slug"),
                "title": entry.get("title"),
                "content": entry.get("content"),
            }
            for entry in branch_chain
        ],
    ]
    question_count = max(2, len(source_branch_path) * 2)
    tags = _branch_semantic_tags(branch_chain)
    rules = [
        str(entry.get("content") or entry.get("title") or "")
        for entry in entries_by_slug.values()
        if _entry_type(entry) == "rule"
    ]
    generated = build_faq_markdown_from_branch({
        "persona": persona,
        "brand": by_type.get("brand"),
        "briefing": by_type.get("briefing"),
        "campaign": by_type.get("campaign"),
        "audience": by_type.get("audience"),
        "product": by_type.get("product"),
        "offer": by_type.get("offer"),
        "copy": by_type.get("copy"),
        "rules": rules,
        "semantic_tags": tags,
        "question_count": question_count,
        "language": "pt-BR",
        "output_format": "markdown",
        "source_branch_path": source_branch_path,
    })
    return _normalize_plan_entry({
        "content_type": "faq",
        "title": generated["title"],
        "slug": _dedupe_slug(generated["slug"], used),
        "status": generated["status"],
        "content": generated["body_markdown"],
        "tags": list(dict.fromkeys(["faq", "golden-dataset", *tags])),
        "metadata": {
            "parent_slug": str(parent.get("slug")),
            "faq_document_type": "golden_dataset",
            "golden_dataset": True,
            "question_count": generated["question_count"],
            "source_branch_path": generated["source_branch_path"],
            "terminal_parent_type": _entry_type(parent),
            "terminal_parent_slug": parent.get("slug"),
            "pending_validation": True,
        },
    })


def _build_grouped_faq_golden_dataset_entry(parent: dict, entries: list[dict], used: set[str], session: dict) -> dict:
    persona_slug = str((session.get("persona_slug") or (session.get("classification") or {}).get("persona_slug") or "persona")).strip()
    source_context = [
        {"content_type": "persona", "slug": persona_slug, "title": persona_slug, "content": f"Persona {persona_slug}"},
        *[
            {
                "content_type": _entry_type(entry),
                "slug": entry.get("slug"),
                "title": entry.get("title"),
                "content": entry.get("content"),
            }
            for entry in entries
            if _entry_type(entry) != "faq"
        ],
    ]
    tags = _branch_semantic_tags([entry for entry in entries if _entry_type(entry) != "faq"])
    rules = [
        str(entry.get("content") or entry.get("title") or "")
        for entry in entries
        if _entry_type(entry) == "rule"
    ]
    question_count = max(8, min(18, len(source_context) + 6))
    generated = build_faq_markdown_from_branch({
        "persona": source_context[0],
        "rules": rules,
        "semantic_tags": tags,
        "question_count": question_count,
        "language": "pt-BR",
        "output_format": "markdown",
        "source_branch_path": source_context,
    })
    return _normalize_plan_entry({
        "content_type": "faq",
        "title": generated["title"],
        "slug": _dedupe_slug("faq-atendimento-agrupado", used),
        "status": generated["status"],
        "content": generated["body_markdown"],
        "tags": list(dict.fromkeys(["faq", "golden-dataset", "grouped", *tags]))[:24],
        "metadata": {
            "parent_slug": str(parent.get("slug")),
            "faq_document_type": "grouped_markdown",
            "golden_dataset": True,
            "grouped_faq": True,
            "question_count": generated["question_count"],
            "source_context_slugs": [str(entry.get("slug")) for entry in entries if entry.get("slug") and _entry_type(entry) != "faq"],
            "terminal_parent_type": _entry_type(parent),
            "terminal_parent_slug": parent.get("slug"),
            "pending_validation": True,
        },
    })


def _terminal_faq_parents(entries: list[dict]) -> list[dict]:
    rules = [entry for entry in entries if _entry_type(entry) == "rule" and entry.get("slug")]
    if rules:
        return rules[:1]
    copies = [entry for entry in entries if _entry_type(entry) == "copy" and entry.get("slug")]
    if copies:
        return copies[:1]
    offers = [entry for entry in entries if _entry_type(entry) == "offer" and entry.get("slug")]
    if offers:
        return offers[:1]
    return [entry for entry in entries if _entry_type(entry) == "product" and entry.get("slug")][:1]


def _has_real_faq_content(entry: dict) -> bool:
    """True for a `faq` entry already carrying a generated question/answer
    pair (services.faq_bulk_generator, via sofia_tools.tool_generate_faq_
    from_branch) -- as opposed to the older placeholder shape this function
    otherwise consolidates into one grouped golden-dataset entry. Without
    this check, a real generated FAQ gets silently deleted the next time the
    plan is renormalized (every _commit call), because the code below used
    to treat ALL `faq` entries as disposable placeholders."""
    metadata = entry.get("metadata") or {}
    return bool(str(metadata.get("question") or "").strip() and str(metadata.get("answer") or "").strip())


def _ensure_faq_golden_datasets_by_branch(plan: dict, session: dict) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    if any(_entry_type(entry) == "faq" and _has_real_faq_content(entry) for entry in entries):
        # At least one FAQ already has real generated content (services.
        # faq_bulk_generator) -- leave the plan untouched instead of
        # replacing it with a content-free grouped placeholder.
        return
    if _requested_variation_count(session, "faq", 0) <= 0 and not any(_entry_type(entry) == "faq" for entry in entries):
        return
    non_faq_entries = [entry for entry in entries if _entry_type(entry) != "faq"]
    parents = _terminal_faq_parents(non_faq_entries)
    if not parents:
        plan["entries"] = non_faq_entries
        return
    used = {str(entry.get("slug")) for entry in entries if entry.get("slug")}
    golden_entries = [_build_grouped_faq_golden_dataset_entry(parents[0], non_faq_entries, used, session)]
    plan["entries"] = [*non_faq_entries, *golden_entries]
    plan["faq_count_policy"] = "grouped"
    plan["faq_parent_type"] = _entry_type(parents[0]) if parents else "copy"
    plan["faq_count_per_parent"] = 1


def _ensure_faqs_per_parent(plan: dict, session: dict) -> None:
    _ensure_faq_golden_datasets_by_branch(plan, session)


def _dedupe_plan_entries(plan: dict) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    used: set[str] = set()
    remap: dict[str, str] = {}
    for index, entry in enumerate(entries, 1):
        old_slug = str(entry.get("slug") or _slug_for_plan_entry(entry.get("title") or f"entry-{index}"))
        if _entry_type(entry) == "product" and "-audience-" in old_slug:
            old_slug = old_slug.split("-audience-", 1)[0]
        new_slug = _dedupe_slug(old_slug, used)
        original_slug = str(entry.get("slug") or "")
        if new_slug != original_slug:
            if original_slug:
                remap[original_slug] = new_slug
            remap[old_slug] = new_slug
            entry["slug"] = new_slug
        meta = _entry_metadata(entry)
        meta.setdefault("plan_entry_id", f"plan-{index:03d}-{new_slug}")
        meta.setdefault("client_id", meta["plan_entry_id"])
    if remap:
        for entry in entries:
            parent = _entry_parent_slug(entry)
            if parent in remap:
                _set_entry_parent_slug(entry, remap[parent])
        for link in plan.get("links") or []:
            if not isinstance(link, dict):
                continue
            if link.get("source_slug") in remap:
                link["source_slug"] = remap[link["source_slug"]]
            if link.get("target_slug") in remap:
                link["target_slug"] = remap[link["target_slug"]]
    plan["entries"] = entries


def _clone_plan_entry(template: dict, *, title: str, slug: str, parent_slug: str, content: str, tags: Optional[list[str]] = None) -> dict:
    clone = _normalize_plan_entry(template)
    clone["title"] = title
    clone["slug"] = slug
    clone["content"] = content
    clone["status"] = "pendente_validacao"
    clone["tags"] = [str(tag).strip() for tag in (tags or clone.get("tags") or []) if str(tag).strip()]
    metadata = dict(clone.get("metadata") or {})
    metadata["parent_slug"] = parent_slug
    metadata["branch_generated"] = True
    clone["metadata"] = metadata
    return clone


def _build_links_from_parent_slugs(plan: dict) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    slug_to_type = {str(entry.get("slug")): _entry_type(entry) for entry in entries if entry.get("slug")}
    parent_by_target = {
        str(entry.get("slug")): str(_entry_parent_slug(entry) or "")
        for entry in entries
        if entry.get("slug") and _entry_parent_slug(entry)
    }
    deduped: dict[tuple[str, str], dict] = {}
    for link in plan.get("links") or []:
        if not isinstance(link, dict):
            continue
        source_slug = str(link.get("source_slug") or "").strip()
        target_slug = str(link.get("target_slug") or "").strip()
        if not source_slug or not target_slug:
            continue
        declared_parent = parent_by_target.get(target_slug)
        if declared_parent and declared_parent != source_slug:
            continue
        deduped[(source_slug, target_slug)] = {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "relation_type": str(link.get("relation_type") or "").strip() or _relation_type_for_parent(
                slug_to_type.get(source_slug, ""),
                slug_to_type.get(target_slug, ""),
            ),
            **({"graph_layer": link.get("graph_layer")} if link.get("graph_layer") else {}),
            **({"primary_tree": link.get("primary_tree")} if "primary_tree" in link else {}),
        }
    for entry in entries:
        source_slug = _entry_parent_slug(entry)
        target_slug = str(entry.get("slug") or "").strip()
        if not source_slug or not target_slug or source_slug == target_slug:
            continue
        deduped[(source_slug, target_slug)] = {
            "source_slug": source_slug,
            "target_slug": target_slug,
            "relation_type": _relation_type_for_parent(slug_to_type.get(source_slug, ""), _entry_type(entry)),
        }
    plan["links"] = list(deduped.values())


def _add_grouped_commercial_flow_links(plan: dict) -> None:
    entries = [entry for entry in (plan.get("entries") or []) if isinstance(entry, dict)]
    rules = [entry for entry in entries if _entry_type(entry) == "rule" and entry.get("slug")]
    rule_slug = str(rules[0].get("slug")) if rules else ""
    deduped: dict[tuple[str, str], dict] = {}
    for link in plan.get("links") or []:
        if not isinstance(link, dict):
            continue
        source_slug = str(link.get("source_slug") or "").strip()
        target_slug = str(link.get("target_slug") or "").strip()
        if source_slug and target_slug:
            deduped[(source_slug, target_slug)] = link
    for copy in [entry for entry in entries if _entry_type(entry) == "copy" and entry.get("slug")]:
        copy_slug = str(copy.get("slug"))
        for offer_slug in _entry_metadata(copy).get("grouped_offer_slugs") or []:
            offer_slug = str(offer_slug or "").strip()
            if not offer_slug or (offer_slug, copy_slug) in deduped:
                continue
            deduped[(offer_slug, copy_slug)] = {
                "source_slug": offer_slug,
                "target_slug": copy_slug,
                "relation_type": "supports_copy",
                "graph_layer": "commercial_grouping",
                "primary_tree": False,
            }
        if rule_slug and (copy_slug, rule_slug) not in deduped:
            deduped[(copy_slug, rule_slug)] = {
                "source_slug": copy_slug,
                "target_slug": rule_slug,
                "relation_type": "contains",
                "graph_layer": "commercial_grouping",
                "primary_tree": False,
            }
    plan["links"] = list(deduped.values())


def _separate_plan_edges(plan: dict) -> None:
    primary_edges = []
    secondary_edges: list[dict[str, Any]] = []
    for link in plan.get("links") or []:
        if not isinstance(link, dict):
            continue
        edge = {
            "source_slug": str(link.get("source_slug") or "").strip(),
            "target_slug": str(link.get("target_slug") or "").strip(),
            "relation_type": str(link.get("relation_type") or "contains").strip() or "contains",
            "graph_layer": str(link.get("graph_layer") or "primary_tree"),
            "primary_tree": bool(link.get("primary_tree", True)),
        }
        if not edge["source_slug"] or not edge["target_slug"]:
            continue
        if edge["primary_tree"]:
            primary_edges.append(edge)
        else:
            secondary_edges.append(edge)
    for entry in plan.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug") or "").strip()
        if not slug:
            continue
        for tag in entry.get("tags") or []:
            tag_slug = _slug_for_plan_entry(str(tag))
            if not tag_slug:
                continue
            secondary_edges.append({
                "source_slug": slug,
                "target_slug": f"tag-{tag_slug}",
                "relation_type": "has_tag",
                "graph_layer": "semantic_tags",
                "primary_tree": False,
                "visual_hidden": True,
            })
    plan["primary_tree_edges"] = primary_edges
    plan["secondary_semantic_edges"] = secondary_edges
    plan.setdefault("rag_edges", [])
    plan.setdefault("asset_gallery_edges", [])
    plan.setdefault("debug_edges", [])


def _base_product_slug(entry: dict) -> str:
    meta = _entry_metadata(entry)
    return str(meta.get("fractal_base_slug") or entry.get("slug") or "").strip()


def _fractal_base_product_slug(entry: dict) -> str:
    raw = _base_product_slug(entry)
    raw = raw.split("-audience-", 1)[0]
    return re.sub(r"-branch-\d+$", "", raw)


def _clone_product_scoped_entry(template: dict, *, product: dict, parent_slug: str, suffix: Optional[str] = None) -> dict:
    base_slug = str(template.get("slug") or _slug_for_plan_entry(template.get("title") or "entry"))
    product_slug = str(product.get("slug") or "")
    clone_slug = f"{base_slug}-{product_slug}" if product_slug and product_slug not in base_slug else base_slug
    if suffix:
        clone_slug = f"{clone_slug}-{suffix}"
    clone = _clone_plan_entry(
        template,
        title=str(template.get("title") or product.get("title") or "Conhecimento"),
        slug=clone_slug,
        parent_slug=parent_slug,
        content=str(template.get("content") or template.get("title") or ""),
        tags=template.get("tags") or [],
    )
    meta = _entry_metadata(clone)
    meta["fractal_base_slug"] = str(template.get("slug") or "")
    meta["fractal_product_slug"] = product_slug
    return clone


def _expand_copies_for_products(entries: list[dict], products: list[dict], product_expansions: dict[str, list[str]]) -> None:
    if not products:
        return
    product_by_slug = {str(product.get("slug")): product for product in products if product.get("slug")}
    entry_by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
    copies = [entry for entry in list(entries) if _entry_type(entry) == "copy"]
    if not copies:
        return

    remove: set[str] = set()
    clones: list[dict] = []
    for copy in copies:
        copy_slug = str(copy.get("slug") or "")
        parent_slug = _entry_parent_slug(copy)
        if _entry_type(entry_by_slug.get(parent_slug or "", {})) == "offer":
            continue
        parent_product = product_by_slug.get(parent_slug or "")
        if parent_product:
            continue

        scoped_products: list[dict] = []
        if parent_slug and parent_slug in product_expansions:
            scoped_products = [
                product_by_slug[slug]
                for slug in product_expansions[parent_slug]
                if slug in product_by_slug
            ]
        if not scoped_products:
            picked = _best_parent_by_slug(copy, products)
            if picked and picked.get("slug"):
                scoped_products = [picked]
            elif len(products) == 1:
                scoped_products = [products[0]]
            else:
                scoped_products = list(products)

        if not scoped_products:
            continue
        if len(scoped_products) == 1 and not (parent_slug and parent_slug in product_expansions and len(product_expansions[parent_slug]) > 1):
            _set_entry_parent_slug(copy, str(scoped_products[0].get("slug") or ""))
            continue
        remove.add(copy_slug)
        for product in scoped_products:
            parent = str(product.get("slug") or "")
            clone = _clone_product_scoped_entry(copy, product=product, parent_slug=parent)
            clones.append(clone)

    if remove:
        entries[:] = [entry for entry in entries if str(entry.get("slug") or "") not in remove]
        entries.extend(clones)


def _normalize_sofia_knowledge_plan(plan: dict, session: dict) -> dict:
    normalized = dict(plan or {})
    normalized["source"] = str(normalized.get("source") or _source_url_from_context(str(session.get("context") or "")) or "session").strip()
    normalized["persona_slug"] = str(normalized.get("persona_slug") or (session.get("classification") or {}).get("persona_slug") or "global").strip()
    normalized["validation_policy"] = str(normalized.get("validation_policy") or "human_validation_required").strip()
    raw_tree_mode = str(normalized.get("tree_mode") or "").strip()
    if raw_tree_mode not in {"pyramidal", "single_branch"}:
        raw_tree_mode = "pyramidal"
    if raw_tree_mode == "single_branch":
        raw_tree_mode = "pyramidal"
    normalized["tree_mode"] = raw_tree_mode
    normalized["branch_policy"] = "top_down_pyramidal"
    normalized["faq_count_policy"] = _normalize_count_policy(normalized, session, "faq")
    if normalized["faq_count_policy"] == "per_branch":
        normalized["faq_count_policy"] = "grouped"
    normalized["faq_parent_type"] = str(normalized.get("faq_parent_type") or "copy").strip() or "copy"
    normalized["faq_count_per_parent"] = max(0, int(normalized.get("faq_count_per_parent") or 1))
    normalized["asset_count_policy"] = _normalize_count_policy(normalized, session, "asset")
    normalized["asset_parent_type"] = str(normalized.get("asset_parent_type") or "product").strip() or "product"
    normalized["asset_count_per_parent"] = max(0, int(normalized.get("asset_count_per_parent") or _requested_variation_count(session, "asset", 0)))
    default_copy_policy = "per_offer" if _explicit_copy_per_offer_requested(session) else "per_product_context"
    normalized["copy_policy"] = str(normalized.get("copy_policy") or default_copy_policy).strip() or default_copy_policy
    if normalized["copy_policy"] == "per_offer" and not _explicit_copy_per_offer_requested(session):
        normalized["copy_policy"] = "per_product_context"

    entries = [_normalize_plan_entry(entry) for entry in (normalized.get("entries") or []) if isinstance(entry, dict)]
    # The persona is the implicit rendered/session root. If the LLM includes a
    # persona entry, it becomes a duplicate terminal card in the preview
    # because children are normalized to parent "self".
    root_persona_slug = _slug_for_plan_entry(str(normalized.get("persona_slug") or ""))
    entries = [
        entry for entry in entries
        if not (
            _entry_type(entry) == "persona"
            and (
                not root_persona_slug
                or _slug_for_plan_entry(str(entry.get("slug") or entry.get("title") or "")) == root_persona_slug
            )
        )
    ]
    normalized["entries"] = entries
    _normalize_plan_parent_slugs(normalized, normalized["persona_slug"])
    _repair_canonical_parent_slugs(normalized)

    # Root scaffolding: persona -> brand? -> briefing -> campaign? -> audience.
    # A briefing without brand is a valid first real node below persona; never
    # leave it pointing to loose sentinels such as "global".
    brands = [entry for entry in entries if _entry_type(entry) == "brand"]
    briefings = [entry for entry in entries if _entry_type(entry) == "briefing"]
    campaigns = [entry for entry in entries if _entry_type(entry) == "campaign"]
    audiences = [entry for entry in entries if _entry_type(entry) == "audience"]
    root_brand = brands[0] if brands else None
    root_briefing = briefings[0] if briefings else None
    root_campaign = campaigns[0] if campaigns else None

    # Legacy scaffolding (auto-create root briefing if missing, force campaign
    # under briefing, etc.) only when NOT in canonical mode. With Sofia tools
    # ON the agent is in charge of creating exactly what the operator asked.
    if not _sofia_tools_enabled():
        if root_briefing is None:
            title = _knowledge_plan_title_from_session(session)
            root_briefing = _normalize_plan_entry({
                "content_type": "briefing",
                "title": title,
                "slug": _slug_for_plan_entry(title),
                "status": "pendente_validacao",
                "content": f"Briefing operacional para {title}. Fonte principal: {normalized['source']}.",
                "tags": ["briefing", normalized["persona_slug"]],
                "metadata": {},
            })
            entries.insert(0, root_briefing)

        if root_brand and _entry_parent_slug(root_brand) in {None, "", "global", "root", "persona"}:
            _set_entry_parent_slug(root_brand, "self")
        briefing_parent = _entry_parent_slug(root_briefing) if root_briefing else None
        if root_briefing and (
            briefing_parent in {None, "", "global", "root", "persona"}
            or (root_brand and briefing_parent == "self")
        ):
            _set_entry_parent_slug(root_briefing, str((root_brand or {}).get("slug") or "self"))
        if root_campaign and not _entry_parent_slug(root_campaign):
            _set_entry_parent_slug(root_campaign, root_briefing["slug"])
        for audience in audiences:
            if not _entry_parent_slug(audience):
                _set_entry_parent_slug(audience, (root_campaign or root_briefing)["slug"])

    # Canonical mode (Janela 2) — when SOFIA_TOOLS_ENABLED is on, the agent
    # creates everything via deterministic tools. Skip auto-expansion (product
    # cloning per audience, copy duplication per product, automatic rule/FAQ)
    # and let normalize stay strictly a cleanup pass.
    canonical_mode = _sofia_tools_enabled()

    # Product branches must live under each audience. If products are still generic,
    # clone them once per audience to create the fractal top-down structure.
    products = [entry for entry in list(entries) if _entry_type(entry) == "product"]
    audience_map = {str(entry.get("slug")): entry for entry in entries if _entry_type(entry) == "audience" and entry.get("slug")}
    product_expansions: dict[str, list[str]] = {}
    if audience_map and not canonical_mode:
        expanded_products: list[dict] = []
        remove_slugs: set[str] = set()
        existing_product_slugs = {str(product.get("slug")) for product in products if product.get("slug")}
        audience_index = {slug: index for index, slug in enumerate(audience_map.keys(), 1)}
        distribute_without_cloning = False
        audience_order = list(audience_map.keys())
        audience_slugs = set(audience_map.keys())
        coverage_by_base: dict[str, set[str]] = {}
        product_slugs_by_base: dict[str, list[str]] = {}
        for product in products:
            parent_slug = _entry_parent_slug(product)
            if parent_slug not in audience_map:
                continue
            base = _fractal_base_product_slug(product)
            coverage_by_base.setdefault(base, set()).add(str(parent_slug))
            if product.get("slug"):
                product_slugs_by_base.setdefault(base, []).append(str(product.get("slug")))
        complete_bases = {
            base for base, covered in coverage_by_base.items()
            if audience_slugs and audience_slugs.issubset(covered)
        }
        for product_index, product in enumerate(products):
            parent_slug = _entry_parent_slug(product)
            parent_type = _entry_type(audience_map.get(parent_slug, {})) if parent_slug else ""
            base_slug = str(product.get("slug"))
            branch_base_slug = _fractal_base_product_slug(product)
            if parent_type == "audience":
                if branch_base_slug in complete_bases:
                    product_expansions[branch_base_slug] = list(dict.fromkeys(product_slugs_by_base.get(branch_base_slug, [base_slug])))
                    product_expansions.setdefault(base_slug, [base_slug])
                    continue
                product_expansions.setdefault(base_slug, [base_slug])
                if len(audience_map) == 1 or distribute_without_cloning:
                    continue
                for audience_slug, audience in audience_map.items():
                    if audience_slug == parent_slug:
                        continue
                    clone_slug = _dedupe_slug(f"{branch_base_slug}-branch-{audience_index.get(audience_slug, 1)}", existing_product_slugs)
                    clone = _clone_plan_entry(
                        product,
                        title=product["title"],
                        slug=clone_slug,
                        parent_slug=audience_slug,
                        content=str(product.get("content") or ""),
                        tags=product.get("tags") or [],
                    )
                    clone_meta = _entry_metadata(clone)
                    clone_meta["fractal_base_slug"] = base_slug
                    expanded_products.append(clone)
                    product_expansions.setdefault(base_slug, [base_slug]).append(clone_slug)
                continue
            if len(audience_map) == 1:
                _set_entry_parent_slug(product, next(iter(audience_map.keys())))
                product_expansions.setdefault(base_slug, [base_slug])
                continue
            if distribute_without_cloning and audience_order:
                _set_entry_parent_slug(product, audience_order[product_index % len(audience_order)])
                product_expansions.setdefault(base_slug, [base_slug])
                continue
            remove_slugs.add(base_slug)
            for audience_slug, audience in audience_map.items():
                clone_slug = _dedupe_slug(f"{branch_base_slug}-branch-{audience_index.get(audience_slug, 1)}", existing_product_slugs)
                clone = _clone_plan_entry(
                    product,
                    title=product["title"],
                    slug=clone_slug,
                    parent_slug=audience_slug,
                    content=str(product.get("content") or ""),
                    tags=product.get("tags") or [],
                )
                clone_meta = _entry_metadata(clone)
                clone_meta["fractal_base_slug"] = base_slug
                expanded_products.append(clone)
                product_expansions.setdefault(base_slug, []).append(clone_slug)
        if remove_slugs:
            entries[:] = [entry for entry in entries if str(entry.get("slug")) not in remove_slugs]
        if expanded_products:
            entries.extend(expanded_products)

    # Copies are grouped per audience->product context by default. FAQ is a
    # single grouped markdown document and sits after the governing rule.
    entries_by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
    products = [entry for entry in entries if _entry_type(entry) == "product"]
    normalized["entries"] = entries
    if not canonical_mode:
        # Legacy expansion machinery (Janela 2 removed these on the canonical
        # path). Kept behind the flag so non-Sofia callers keep working.
        _ensure_offer_entries(normalized, session)
        entries = [entry for entry in (normalized.get("entries") or []) if isinstance(entry, dict)]
        entries_by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
        products = [entry for entry in entries if _entry_type(entry) == "product"]
        _expand_copies_for_products(entries, products, product_expansions)
        normalized["entries"] = entries
        _ensure_copies_for_offers(normalized, session)
        _reparent_copies_to_offers(normalized, session)
        entries = [entry for entry in (normalized.get("entries") or []) if isinstance(entry, dict)]
        entries_by_slug = {str(entry.get("slug")): entry for entry in entries if entry.get("slug")}
        normalized["entries"] = entries
        _ensure_governing_rule(normalized, session)
        _ensure_faqs_per_parent(normalized, session)
    _dedupe_plan_entries(normalized)
    _auto_infer_parent_slugs(normalized)
    _normalize_plan_parent_slugs(normalized, normalized["persona_slug"])
    _build_links_from_parent_slugs(normalized)
    if not canonical_mode:
        _add_grouped_commercial_flow_links(normalized)
    _separate_plan_edges(normalized)
    normalized["summary"] = summarize_normalized_plan(normalized)
    return normalized


def _normalize_live_session_plan(plan: dict, session: dict) -> dict:
    normalized = dict(plan or {})
    normalized["source"] = str(
        normalized.get("source")
        or session.get("source_url")
        or _source_url_from_context(str(session.get("context") or ""))
        or "session"
    ).strip()
    normalized["persona_slug"] = str(
        normalized.get("persona_slug")
        or session.get("persona_slug")
        or (session.get("classification") or {}).get("persona_slug")
        or "global"
    ).strip()
    normalized["validation_policy"] = str(normalized.get("validation_policy") or "human_validation_required").strip()
    normalized["tree_mode"] = str(normalized.get("tree_mode") or "pyramidal").strip() or "pyramidal"
    normalized["branch_policy"] = str(normalized.get("branch_policy") or "top_down_pyramidal").strip() or "top_down_pyramidal"
    normalized["faq_count_policy"] = _normalize_count_policy(normalized, session, "faq")
    if normalized["faq_count_policy"] == "per_branch":
        normalized["faq_count_policy"] = "grouped"
    normalized["faq_parent_type"] = str(normalized.get("faq_parent_type") or "copy").strip() or "copy"
    normalized["faq_count_per_parent"] = max(0, int(normalized.get("faq_count_per_parent") or 1))
    normalized["asset_count_policy"] = _normalize_count_policy(normalized, session, "asset")
    normalized["asset_parent_type"] = str(normalized.get("asset_parent_type") or "product").strip() or "product"
    normalized["asset_count_per_parent"] = max(0, int(normalized.get("asset_count_per_parent") or _requested_variation_count(session, "asset", 0)))
    default_copy_policy = "per_offer" if _explicit_copy_per_offer_requested(session) else "per_product_context"
    normalized["copy_policy"] = str(normalized.get("copy_policy") or default_copy_policy).strip() or default_copy_policy
    if normalized["copy_policy"] == "per_offer" and not _explicit_copy_per_offer_requested(session):
        normalized["copy_policy"] = "per_product_context"
    normalized["entries"] = [
        _normalize_plan_entry(entry)
        for entry in (normalized.get("entries") or [])
        if isinstance(entry, dict)
    ]
    links: list[dict[str, str]] = []
    for link in normalized.get("links") or []:
        if not isinstance(link, dict):
            continue
        source_slug = str(link.get("source_slug") or "").strip()
        target_slug = str(link.get("target_slug") or "").strip()
        if not source_slug or not target_slug:
            continue
        links.append({
            "source_slug": source_slug,
            "target_slug": target_slug,
            "relation_type": str(link.get("relation_type") or "contains").strip() or "contains",
        })
    normalized["links"] = links
    return normalized


def _rewrite_visible_plan_summary(message: str, plan_payload: Optional[dict]) -> str:
    if not message or not isinstance(plan_payload, dict):
        return message
    if "entries" not in plan_payload:
        return message
    summary = summarize_normalized_plan(plan_payload)
    counts = summary.get("current_block_counts") or {}
    if not plan_payload.get("entries"):
        return "Status: bloqueado\nMotivo: Estrutura ainda não gerada.\nAção: corrigir branch ou responder campo pendente."
    expansion = _expansion_summary(plan_payload)
    faq = expansion.get("faq") or {}
    asset = expansion.get("asset") or {}
    summary_line = "\n".join([
        "Status: plano gerado",
        (
            f"Resumo: briefing {counts.get('briefing', 0)}, público {counts.get('audience', 0)}, "
            f"produto {counts.get('product', 0)}, oferta {counts.get('offer', 0)}, copy {counts.get('copy', 0)}, "
            f"FAQ {counts.get('faq', 0)}, asset {counts.get('asset', 0)}, regra {counts.get('rule', 0)}"
        ),
        (
            f"Política: árvore piramidal; FAQ por {faq.get('parent_type') or 'copy'}; "
            f"Asset por {asset.get('parent_type') or 'parent'}"
        ),
        "Pendências bloqueantes: nenhuma",
        "Ação: revisar preview",
    ])
    link_count = len(plan_payload.get("links") or [])
    if re.search(r"(?im)^Conex\S*:\s*\d+\s+edges no plano\s*$", message):
        updated = re.sub(
            r"(?im)^Conex\S*:\s*\d+\s+edges no plano\s*$",
            f"Conexões: {link_count} edges no plano",
            message,
        )
        return updated if summary_line in updated else f"{updated}\n{summary_line}"
    if "Plano pronto. Clique em **Salvar** para persistir." in message:
        return re.sub(r"(?s)Plano pronto\. Clique em \*\*Salvar\*\* para persistir\.", summary_line, message)
    return summary_line


def _serialize_session(session: dict) -> dict:
    data = json.loads(json.dumps(session, default=str))
    raw = session.get("classification", {}).get("file_bytes")
    if isinstance(raw, (bytes, bytearray)):
        data.setdefault("classification", {})["file_bytes_b64"] = base64.b64encode(raw).decode("ascii")
        data["classification"]["file_bytes"] = None
    return data


def _save_session(session: dict) -> None:
    try:
        supabase_client.upsert_kb_intake_session(_serialize_session(session))
    except Exception:
        pass


def _load_session(session_id: str) -> Optional[dict]:
    try:
        session = supabase_client.get_kb_intake_session(session_id)
        if not session:
            return None
        b64 = session.get("classification", {}).pop("file_bytes_b64", None)
        if b64:
            session["classification"]["file_bytes"] = base64.b64decode(b64)
        _sessions[session_id] = session
        return session
    except Exception:
        return None


def _get_session(session_id: str) -> Optional[dict]:
    session = _sessions.get(session_id) or _load_session(session_id)
    if session:
        session.setdefault("telemetry_transcript", [])
        session.setdefault("telemetry_flags", {"dialog_started_emitted": False})
    return session


def _truncate(value: Any, limit: int = _EVENT_PREVIEW_LIMIT) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _metrics_from_session(session: dict) -> dict[str, Any]:
    transcript = session.get("telemetry_transcript") or []
    user_turns = sum(1 for turn in transcript if turn.get("role") == "user")
    assistant_turns = sum(1 for turn in transcript if turn.get("role") == "assistant")
    messages = session.get("messages") or []
    return {
        "n_user_turns": user_turns,
        "n_assistant_turns": assistant_turns,
        "n_total_turns": len(transcript),
        "message_count": len(messages),
        "has_plan": any(
            msg.get("role") == "assistant" and "<knowledge_plan>" in str(msg.get("content") or "")
            for msg in messages
        ),
        "has_file": bool(session.get("classification", {}).get("file_ext")),
        "has_crawler_capture": bool(session.get("crawler_captures")),
    }


def _session_identity_payload(session: dict) -> dict[str, Any]:
    cls = session.get("classification") or {}
    persona_slug = session.get("persona_slug") or cls.get("persona_slug")
    return {
        "session_id": session.get("id"),
        "agent_key": session.get("agent_key"),
        "agent_name": session.get("agent_name"),
        "model": session.get("model"),
        "persona_slug": persona_slug,
        "content_type": cls.get("content_type"),
        "title": cls.get("title"),
        "stage": session.get("stage"),
        "source_url": session.get("source_url"),
        "initial_block_counts": session.get("initial_block_counts"),
        "current_block_counts": session.get("current_block_counts"),
        "tree_mode": ((session.get("knowledge_plan") or {}).get("tree_mode") if isinstance(session.get("knowledge_plan"), dict) else None),
        "branch_policy": ((session.get("knowledge_plan") or {}).get("branch_policy") if isinstance(session.get("knowledge_plan"), dict) else None),
    }


def _build_event_payload(
    session: dict,
    *,
    status: str,
    result: Optional[dict] = None,
    transcript: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        **_session_identity_payload(session),
        "status": status,
        "metrics": _metrics_from_session(session),
    }
    if result is not None:
        payload["result"] = result
    if transcript:
        payload["transcript"] = session.get("telemetry_transcript") or []
    if extra:
        payload.update(extra)
    return payload


def _emit_kb_event(
    event_type: str,
    *,
    session: dict,
    source: str,
    status: str,
    result: Optional[dict] = None,
    transcript: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    try:
        payload = _build_event_payload(
            session,
            status=status,
            result=result,
            transcript=transcript,
            extra=extra,
        )
        supabase_client.insert_event(
            {
                "event_type": event_type,
                "payload": payload,
            },
            source=source,
        )
    except Exception as exc:
        try:
            from services import sre_logger
            sre_logger.warn(
                "kb_intake_event",
                f"event emission skipped type={event_type} source={source}: {exc}",
                exc,
            )
        except Exception:
            pass


def _append_transcript_turn(
    session: dict,
    *,
    role: str,
    content: str,
    file_attached: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    transcript = session.setdefault("telemetry_transcript", [])
    turn = {
        "turn_index": len(transcript),
        "role": role,
        "message_preview": _truncate(content),
        "message_chars": len(content or ""),
        "file_attached": file_attached,
        "stage": session.get("stage"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        turn.update(extra)
    transcript.append(turn)
    if len(transcript) > _EVENT_TRANSCRIPT_MAX_TURNS:
        session["telemetry_transcript"] = transcript[-_EVENT_TRANSCRIPT_MAX_TURNS:]
        for idx, item in enumerate(session["telemetry_transcript"]):
            item["turn_index"] = idx
        transcript = session["telemetry_transcript"]
    return transcript[-1]


def _default_mission_state(initial_context: str) -> dict[str, Any]:
    url = _source_url_from_context(initial_context or "")
    persona = "global"
    obj = "Criar inteligencia de marketing em grafo com evidencias reais."
    blocks = ["briefing", "audience", "product", "copy", "faq"]
    if initial_context:
        m = re.search(r"persona_slug:\s*([a-z0-9_-]+)", initial_context, re.I)
        if m:
            persona = m.group(1).strip().lower()
        m = re.search(r"objetivo:\s*(.+)", initial_context, re.I)
        if m:
            obj = m.group(1).strip()
        found_blocks: list[str] = []
        for line in initial_context.splitlines():
            s = line.strip().lower()
            if not s.startswith("- "):
                continue
            token = s[2:].split(":", 1)[0].strip()
            token = _CONTENT_ALIASES.get(token, token)
            if token in {"brand", "briefing", "campaign", "audience", "product", "entity", "copy", "faq", "rule", "tone", "asset"}:
                found_blocks.append(token)
        if found_blocks:
            blocks = sorted(set(found_blocks), key=found_blocks.index)
    return {
        "persona": persona,
        "objective": obj,
        "source": {"type": "website", "url": url},
        "knowledge_blocks": blocks,
        "requested_outputs": {"models": []},
        "format": "default_intelligence_graph",
        "status": "collecting",
        "evidence_items": [],
        "last_patch": {},
    }


def _context_persona_slug(initial_context: str) -> str | None:
    m = re.search(r"persona_slug:\s*([a-z0-9_-]+)", initial_context or "", re.I)
    if m:
        return m.group(1).strip().lower()
    return None


def _context_objective(initial_context: str) -> str:
    m = re.search(r"objetivo:\s*(.+)", initial_context or "", re.I)
    if m:
        return m.group(1).strip()
    return ""


def _session_matches_resume_candidate(
    session: dict,
    *,
    persona_slug: str | None,
    agent_key: str,
    objective: str,
    source_url: str | None,
) -> bool:
    if (session.get("agent_key") or "sofia") != agent_key:
        return False
    stage = (session.get("stage") or "").lower()
    if stage == "done":
        return False
    candidate_persona = (
        (session.get("classification") or {}).get("persona_slug")
        or (session.get("mission_state") or {}).get("persona")
    )
    if persona_slug and candidate_persona and candidate_persona != persona_slug:
        return False
    if objective:
        candidate_objective = str((session.get("mission_state") or {}).get("objective") or "").strip().lower()
        if candidate_objective and candidate_objective != objective.strip().lower():
            return False
    if source_url:
        candidate_source = str(((session.get("mission_state") or {}).get("source") or {}).get("url") or "").strip().lower()
        if candidate_source and candidate_source != source_url.strip().lower():
            return False
    return True


def _latest_persisted_resume_session(initial_context: str, agent_key: str) -> Optional[dict]:
    persona_slug = _context_persona_slug(initial_context)
    objective = _context_objective(initial_context)
    source_url = _source_url_from_context(initial_context)
    candidates: list[dict] = []
    try:
        for session in supabase_client.list_kb_intake_sessions(limit=500):
            if not _session_matches_resume_candidate(
                session,
                persona_slug=persona_slug,
                agent_key=agent_key,
                objective=objective,
                source_url=source_url,
            ):
                continue
            candidates.append(session)
    except Exception:
        return None
    if not candidates:
        return None
    return candidates[0]


def _resume_summary_from_payload(payload: dict[str, Any]) -> str:
    transcript = payload.get("transcript") or []
    last_assistant = next(
        (turn.get("message_preview") for turn in reversed(transcript) if turn.get("role") == "assistant" and turn.get("message_preview")),
        "",
    )
    parts = [
        f"Persona: {payload.get('persona_slug') or 'nao informada'}",
        f"Tipo: {payload.get('content_type') or 'nao definido'}",
        f"Titulo: {payload.get('title') or 'sem titulo'}",
    ]
    if last_assistant:
        parts.append(f"Ultima resposta: {last_assistant}")
    return "\n".join(parts)


def _latest_persisted_resume(initial_context: str, agent_key: str) -> Optional[dict[str, Any]]:
    persona_slug = _context_persona_slug(initial_context)
    source_url = (_source_url_from_context(initial_context) or "").strip().lower()
    try:
        events = supabase_client.get_events(limit=20, event_type="kb_intake_dialog_completed")
    except Exception:
        return None
    for event in events or []:
        payload = event.get("payload") or {}
        if (payload.get("agent_key") or "sofia") != agent_key:
            continue
        if persona_slug and payload.get("persona_slug") != persona_slug:
            continue
        candidate_source = str(payload.get("source") or "").strip().lower()
        if source_url and candidate_source and candidate_source != source_url:
            continue
        return {
            "resumed_from_session_id": payload.get("session_id"),
            "resume_source": "system_events",
            "resume_summary": _resume_summary_from_payload(payload),
        }
    return None


def _build_resume_metadata(initial_context: str, agent_key: str) -> dict[str, Any]:
    local_session = _latest_persisted_resume_session(initial_context, agent_key)
    if local_session:
        payload = _build_event_payload(local_session, status=str(local_session.get("stage") or "chatting"))
        return {
            "resumed_from_session_id": local_session.get("id"),
            "resume_source": "local_session",
            "resume_summary": _resume_summary_from_payload(payload),
        }
    return _latest_persisted_resume(initial_context, agent_key) or {
        "resumed_from_session_id": None,
        "resume_source": None,
        "resume_summary": "",
    }


def _context_with_resume(initial_context: str, resume_meta: dict[str, Any]) -> str:
    context = (initial_context or "").strip()
    resume_summary = (resume_meta or {}).get("resume_summary") or ""
    if not resume_summary:
        return context
    block = [
        "## Retomada automatica",
        f"source: {resume_meta.get('resume_source')}",
        f"session_id_anterior: {resume_meta.get('resumed_from_session_id') or 'desconhecida'}",
        resume_summary,
    ]
    return "\n\n".join([part for part in [context, "\n".join(block)] if part])


def _session_public_state(session: dict) -> dict[str, Any]:
    prune_unpersisted_asset_readings(session)
    normalized_plan = session.get("normalized_plan") or session.get("knowledge_plan")
    plan_summary = session.get("plan_summary") or (
        summarize_normalized_plan(normalized_plan) if isinstance(normalized_plan, dict) else None
    )
    plan_validation = session.get("plan_validation") or _plan_validation(warnings=session.get("plan_validation_warnings") or [])
    plan_hash = session.get("plan_hash") or (_plan_hash(normalized_plan) if isinstance(normalized_plan, dict) else None)
    plan_state = None
    if isinstance(normalized_plan, dict) and normalized_plan.get("entries"):
        plan_state = {
            "normalized_plan": normalized_plan,
            "validation": plan_validation,
            "summary": plan_summary,
            "plan_hash": plan_hash,
        }
    return {
        "persona_slug": session.get("persona_slug") or (session.get("classification") or {}).get("persona_slug"),
        "persona_id": session.get("persona_id") or ((session.get("mission_state") or {}).get("persona_id")),
        "source_url": session.get("source_url"),
        "mode": session.get("mode") or "legacy",
        "status": session.get("status") or session.get("stage"),
        "initial_block_counts": _normalize_block_counts(session.get("initial_block_counts")),
        "current_block_counts": _normalize_block_counts((plan_summary or {}).get("current_block_counts") or session.get("current_block_counts")),
        "knowledge_plan": normalized_plan,
        "normalized_plan": normalized_plan,
        "plan_state": plan_state,
        "plan_validation": plan_validation,
        "plan_summary": plan_summary,
        "plan_hash": plan_hash,
        "confirmed_plan_hash": session.get("confirmed_plan_hash"),
        "memory_summary": session.get("memory_summary") or "",
        "plan_changed": bool(session.get("plan_changed")),
        "asset_readings": session.get("asset_readings") or [],
        "asset_readings_pruned": int(session.get("asset_readings_pruned") or 0),
        "pre_init_review": session.get("pre_init_review") or None,
        "persona_context": session.get("persona_context") or None,
    }


def _bootstrap_result_payload(session: dict, result: dict[str, Any]) -> dict[str, Any]:
    live_state = _session_public_state(session)
    return {
        "session_id": session["id"],
        "model": session["model"],
        "model_name": AVAILABLE_MODELS.get(session["model"], session["model"]),
        "agent": {
            "key": session.get("agent_key"),
            "name": session.get("agent_name"),
            "role": session.get("agent_role"),
        },
        "welcome": session.get("agent_greeting"),
        "bootstrap_message": result.get("message") or "",
        "classification": result.get("classification") or {},
        "stage": result.get("stage") or session.get("stage"),
        "state": result.get("state") or session.get("mission_state"),
        "resumed_from_session_id": session.get("resumed_from_session_id"),
        "resume_source": session.get("resume_source"),
        "resume_summary": session.get("resume_summary"),
        "knowledge_plan": live_state["knowledge_plan"] or result.get("proposed_plan"),
        "normalized_plan": live_state.get("normalized_plan"),
        "plan_state": live_state.get("plan_state") or result.get("plan_state"),
        "plan_validation": live_state.get("plan_validation"),
        "plan_summary": live_state.get("plan_summary"),
        "plan_hash": live_state.get("plan_hash"),
        "confirmed_plan_hash": live_state.get("confirmed_plan_hash"),
        "current_block_counts": live_state["current_block_counts"],
        "initial_block_counts": live_state["initial_block_counts"],
        "persona_slug": live_state["persona_slug"],
        "persona_id": live_state.get("persona_id"),
        "source_url": live_state["source_url"],
        "memory_summary": live_state["memory_summary"],
        "plan_changed": bool(result.get("plan_changed")),
        "bootstrap_llm": bool(result.get("bootstrap_llm", True)),
        "timings_ms": result.get("timings_ms") or {},
        "pre_init_review": live_state.get("pre_init_review"),
    }


def _resolve_session_persona(persona_slug_or_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    raw = str(persona_slug_or_id or "").strip()
    if not raw:
        return None, None
    try:
        if len(raw) == 36 and raw.count("-") == 4:
            persona = supabase_client.get_persona_by_id(raw) or {}
        else:
            persona = supabase_client.get_persona(raw) or {}
        return persona.get("id") or (raw if len(raw) == 36 and raw.count("-") == 4 else None), persona.get("slug") or raw
    except Exception:
        return (raw if len(raw) == 36 and raw.count("-") == 4 else None), raw


def _deterministic_bootstrap_message(session: dict) -> str:
    counts = _format_block_counts(session.get("current_block_counts"))
    persona = session.get("persona_slug") or "persona selecionada"
    source = session.get("source_url") or "fonte ainda nao informada"
    return (
        f"Sessao da Sofia iniciada para {persona}. "
        f"Plano atual: {counts}. "
        f"Fonte: {source}. "
        "Pode anexar arquivos ou enviar a proxima instrucao."
    )


_PRE_INIT_NODE_TYPES = (
    "brand",
    "briefing",
    "campaign",
    "audience",
    "product",
    "offer",
    "copy",
    "rule",
    "asset",
    "faq",
)


def _load_persona_context(persona_id: Optional[str]) -> dict[str, list[dict]]:
    """Snapshot existing canonical nodes for a persona so the pre-init review
    can recommend reuse instead of asking Sofia to generate duplicates.

    Returns a dict keyed by node_type containing compact dicts. Safe to call
    with persona_id=None (returns empty buckets) and resilient to Supabase
    being unavailable (returns empty buckets in CI/offline)."""
    buckets: dict[str, list[dict]] = {ntype: [] for ntype in _PRE_INIT_NODE_TYPES}
    if not persona_id:
        return buckets
    try:
        rows = supabase_client.list_knowledge_nodes_by_type(
            list(_PRE_INIT_NODE_TYPES),
            persona_id=persona_id,
            limit=400,
        ) or []
    except Exception:
        return buckets
    for row in rows:
        ntype = str(row.get("node_type") or "").lower()
        if ntype not in buckets:
            continue
        buckets[ntype].append({
            "id": row.get("id"),
            "slug": row.get("slug"),
            "title": row.get("title"),
            "tags": row.get("tags") or [],
            "metadata": row.get("metadata") or {},
        })
    return buckets


def build_pre_initialization_review(
    session: dict,
    *,
    persona_context: Optional[dict[str, list[dict]]] = None,
    classification: Optional[dict] = None,
) -> dict[str, Any]:
    """Inspect the persona context and produce a /tree-reference style payload
    that recommends reuse for audience/product/campaign/asset before Sofia
    generates new branches.

    Output shape mirrors the contract documented in CLAUDE.md:
        {
          "persona_context_loaded": bool,
          "existing_nodes_found": { brand[], briefing[], campaign[], audience[],
                                    product[], offer[], copy[], rule[],
                                    asset[], faq[] },
          "recommended_connections": [{type, target_slug, target_title, reason}],
          "new_nodes_needed": [{type, reason}],
          "questions": [string]
        }
    """
    persona_context = persona_context or _load_persona_context(session.get("persona_id"))
    classification = classification or session.get("classification") or {}
    initial_counts = _normalize_block_counts(session.get("initial_block_counts"))
    plan = session.get("knowledge_plan") if isinstance(session.get("knowledge_plan"), dict) else None
    plan_counts = count_blocks_by_type((plan or {}).get("entries") or []) if plan else {}

    existing_nodes_found = {
        ntype: [
            {"slug": item.get("slug"), "title": item.get("title")}
            for item in items
            if item.get("slug")
        ]
        for ntype, items in (persona_context or {}).items()
    }
    has_any_context = any(existing_nodes_found.get(ntype) for ntype in _PRE_INIT_NODE_TYPES)
    recommended: list[dict[str, Any]] = []
    new_nodes_needed: list[dict[str, Any]] = []
    questions: list[str] = []
    session_text = _fold(
        " ".join(
            [
                str(session.get("context") or ""),
                _session_user_text(session),
            ]
        )
    )

    # Audience reuse: if the persona already has audience nodes and the plan
    # has no explicit audience yet, suggest reusing instead of cloning.
    existing_audiences = existing_nodes_found.get("audience") or []
    plan_has_audience = int(plan_counts.get("audience") or 0) > 0
    requested_audience = int(initial_counts.get("audience") or 0) > 0
    explicit_default_audience = bool(
        re.search(
            r"\b(audiencia|audience|publico|publico-alvo)\s+(padrao|default)\b|\bconect(e|ar|ar)\s+a\s+(audiencia|audience)\s+padrao\b",
            session_text,
        )
    )
    if existing_audiences and not plan_has_audience:
        for audience in existing_audiences[:3]:
            recommended.append({
                "type": "audience",
                "target_slug": audience.get("slug"),
                "target_title": audience.get("title"),
                "reason": "audience ja existente para esta persona",
            })
        if requested_audience and not explicit_default_audience:
            questions.append(
                f"Encontrei audiencia ja descrita: {existing_audiences[0].get('title')}. "
                "Deseja conectar nela ou criar uma nova segmentacao?"
            )
        elif not explicit_default_audience:
            questions.append(
                f"A audiencia existente '{existing_audiences[0].get('title')}' atende esta captura ou precisa de nova segmentacao?"
            )

    # Campaign reuse: same idea.
    existing_campaigns = existing_nodes_found.get("campaign") or []
    plan_has_campaign = int(plan_counts.get("campaign") or 0) > 0
    if existing_campaigns and not plan_has_campaign:
        recommended.append({
            "type": "campaign",
            "target_slug": existing_campaigns[0].get("slug"),
            "target_title": existing_campaigns[0].get("title"),
            "reason": "campanha ja cadastrada nesta persona",
        })
        questions.append(
            f"Esta captura deve entrar na campanha existente '{existing_campaigns[0].get('title')}' ou e uma nova campanha?"
        )

    # Group ambiguity: if the operator mentions plural groups without a count,
    # keep the deterministic flow at one group and ask a single clarification
    # only when the transcript is actually ambiguous.
    has_group_word = bool(re.search(r"\bgrupos?\b", session_text))
    has_explicit_group_count = bool(re.search(r"\d+\s*grupos?", session_text))
    singular_group_hint = bool(re.search(r"\bgrupo\s+[a-z0-9]", session_text))
    if has_group_word and not has_explicit_group_count and not singular_group_hint:
        questions.append(
            "O texto menciona grupos sem quantidade explicitada. Você quer criar 2 grupos ou manter 1 grupo com os produtos extraídos?"
        )

    # Asset routing: route by classification.asset_function instead of the
    # generic "product" default. The actual reparent happens in the planner;
    # here we just surface the recommendation for the operator.
    asset_function = str((classification or {}).get("asset_function") or "").lower().strip()
    if asset_function in {"campaign_hero", "campaign_banner"}:
        target = existing_campaigns[0] if existing_campaigns else None
        recommended.append({
            "type": "asset",
            "target_slug": (target or {}).get("slug"),
            "target_title": (target or {}).get("title"),
            "reason": f"asset com funcao {asset_function} deve apoiar a campanha",
            "expected_parent_type": "campaign",
        })
    elif asset_function in {"product_reference"}:
        existing_products = existing_nodes_found.get("product") or []
        target = existing_products[0] if existing_products else None
        recommended.append({
            "type": "asset",
            "target_slug": (target or {}).get("slug"),
            "target_title": (target or {}).get("title"),
            "reason": "asset com funcao product_reference deve apoiar o produto",
            "expected_parent_type": "product",
        })

    # New nodes needed: when the operator requested a count but there is no
    # existing reuse target, flag it.
    for ntype in ("brand", "briefing", "campaign"):
        if int(initial_counts.get(ntype) or 0) > 0 and not (existing_nodes_found.get(ntype) or []):
            new_nodes_needed.append({"type": ntype, "reason": f"persona ainda nao tem {ntype}"})

    return {
        "persona_context_loaded": bool(has_any_context),
        "existing_nodes_found": existing_nodes_found,
        "recommended_connections": recommended,
        "new_nodes_needed": new_nodes_needed,
        "questions": questions,
    }


def _persona_to_slug(raw: str) -> str:
    val = _fold(raw).strip()
    return _PERSONA_ALIASES.get(val, val.replace(" ", "-"))


def _coerce_urlish_value(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.strip("`'\"()[]{}<>.,; ")
    if not raw:
        return None
    if re.match(r"^https?://[^\s/$.?#].[^\s]*$", raw, re.I):
        return raw
    if re.match(r"^(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/[^\s]*)?$", raw, re.I):
        return f"https://{raw}"
    return None


def _persona_from_urlish(value: str) -> str | None:
    site = _coerce_urlish_value(value)
    if not site:
        return None
    host = re.sub(r"^https?://", "", site, flags=re.I).split("/", 1)[0].strip().lower()
    return _PERSONA_DOMAINS.get(host)


def _upsert_model(state: dict, name: str) -> dict:
    models = state.setdefault("requested_outputs", {}).setdefault("models", [])
    for m in models:
        if _fold(m.get("name", "")) == _fold(name):
            return m
    default_qty = state.setdefault("requested_outputs", {}).get("default_products_requested")
    row = {"name": name, "audience": None, "products_requested": default_qty, "fields": []}
    models.append(row)
    return row


def _merge_user_intent(state: dict[str, Any], message: str) -> dict[str, Any]:
    text = message.strip()
    low = _fold(text)
    patch: dict[str, Any] = {}

    m = re.search(r"mudar para\s+([a-z0-9 _-]+?)\s+no site\s+([a-z0-9._/-]+)", low, re.I)
    if m:
        persona = m.group(1).strip()
        site = m.group(2).strip()
        if not site.startswith("http"):
            site = f"https://{site}"
        state["persona"] = _persona_to_slug(persona)
        state["source"] = {"type": "website", "url": site}
        state["objective"] = f"Criar conhecimento de marketing para {persona.title()} a partir de {site}, mantendo os blocos selecionados."
        patch.update({"persona": state["persona"], "source.url": site, "objective": state["objective"]})

    if not patch:
        site = _coerce_urlish_value(text)
        if site:
            state["source"] = {"type": "website", "url": site}
            inferred_persona = _persona_from_urlish(site)
            if inferred_persona and (state.get("persona") in {None, "", "global"}):
                state["persona"] = inferred_persona
                patch["persona"] = inferred_persona
            patch["source.url"] = site

    if "os mesmos" in low:
        patch["knowledge_blocks"] = "preserve_existing"

    m2 = re.search(r"(\d+)\s+produtos?\s+de\s+cada", low)
    if m2:
        qty = int(m2.group(1))
        state.setdefault("requested_outputs", {})["default_products_requested"] = qty
        for model in state.setdefault("requested_outputs", {}).setdefault("models", []):
            model["products_requested"] = qty
            model["fields"] = ["price", "angle", "faq"]
        patch["requested_outputs.products_requested_each"] = qty

    for name in ("juliet", "radar ev", "radar"):
        if name in low:
            canonical = "Radar Ev" if "radar" in name else "Juliet"
            _upsert_model(state, canonical)

    if "street" in low and "juliet" in low:
        row = _upsert_model(state, "Juliet")
        row["audience"] = "Street"
        patch["requested_outputs.models.juliet.audience"] = "Street"
    if ("esportes" in low or "esporte" in low) and "radar" in low:
        row = _upsert_model(state, "Radar Ev")
        row["audience"] = "Esportes"
        patch["requested_outputs.models.radar_ev.audience"] = "Esportes"

    if "faq" in low and "angle" in low:
        for model in state.setdefault("requested_outputs", {}).setdefault("models", []):
            model["fields"] = ["price", "angle", "faq"]

    state["last_patch"] = patch
    return patch


def _mission_summary(state: dict[str, Any]) -> str:
    blocks = ", ".join(state.get("knowledge_blocks") or [])
    models = state.get("requested_outputs", {}).get("models", [])
    model_line = "; ".join(
        f"{m.get('name')} -> {m.get('audience') or 'sem publico'}"
        for m in models
    ) or "sem modelos"
    source = ((state.get("source") or {}).get("url") or "sem fonte")
    persona = state.get("persona") or "sem persona"
    action = "Vou explicar a estrutura que estou montando e só vou pedir confirmação quando houver ambiguidade."
    if models:
        action = (
            "Vou montar a arvore da campanha com Brand -> Briefing -> Campaign -> Audience -> "
            "Product Group -> Product -> Copy -> FAQ, usando os produtos extraidos do site."
        )
    return (
        "Atualizei a missao:\n"
        f"{action}\n"
        f"Persona: {persona}\n"
        f"Fonte: {source}\n"
        f"Blocos mantidos: {blocks}\n"
        f"Modelos: {model_line}\n"
        "Agora vou coletar dados reais do site. Nao vou inventar precos ou FAQs."
    )


def _extract_price(product: dict[str, Any]) -> str:
    prices = product.get("prices") or []
    if prices:
        return str(prices[0])
    if product.get("price"):
        return str(product["price"])
    return ""


def _build_evidence_items(state: dict[str, Any], capture: dict[str, Any]) -> list[dict[str, Any]]:
    products = capture.get("product_candidates") or []
    models = state.get("requested_outputs", {}).get("models", [])
    out: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()
    for req in models:
        model_name = req.get("name") or ""
        audience = req.get("audience") or ""
        matched = [p for p in products if model_name and _fold(model_name) in _fold(str(p.get("title") or ""))]
        limit = int(req.get("products_requested") or 0) or 10
        for p in matched[:limit]:
            out.append({
                "name": p.get("title") or model_name,
                "url": capture.get("final_url") or capture.get("url") or "",
                "price": _extract_price(p),
                "model": model_name,
                "audience": audience,
                "angle": "pendente_validacao",
                "faq": [],
                "evidence": {
                    "source_url": capture.get("url") or "",
                    "captured_at": ts,
                    "confidence": "high" if (capture.get("confidence") or 0) >= 0.72 else "medium" if (capture.get("confidence") or 0) >= 0.45 else "low",
                },
            })
    return out


def create_session(
    model: str = "gpt-4o-mini",
    initial_context: str = "",
    agent_key: str = "sofia",
    initial_state: Optional[dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> dict:
    sid = str(uuid.uuid4())
    agent = get_agent_profile(agent_key)
    created_at = datetime.now(timezone.utc).isoformat()
    resume_meta = _build_resume_metadata(initial_context or "", agent_key if agent_key in AGENT_PROFILES else "sofia")
    initial_state = dict(initial_state or {})
    persona_slug = str(initial_state.get("persona_slug") or _context_persona_slug(initial_context or "") or "").strip().lower()
    persona_id, resolved_persona_slug = _resolve_session_persona(persona_slug)
    if resolved_persona_slug:
        persona_slug = resolved_persona_slug
    source_url = _coerce_urlish_value(str(initial_state.get("source_url") or "")) or _source_url_from_context(initial_context or "")
    mode = str(initial_state.get("mode") or "legacy").strip().lower() or "legacy"
    initial_block_counts = _normalize_block_counts(initial_state.get("initial_block_counts"))
    initial_plan = initial_state.get("knowledge_plan") if isinstance(initial_state.get("knowledge_plan"), dict) else None
    current_block_counts = _counts_from_plan_or_initial(initial_plan, initial_block_counts)
    mission_state = _default_mission_state(initial_context or "")
    if persona_slug:
        mission_state["persona"] = persona_slug
        mission_state["persona_slug"] = persona_slug
    if persona_id:
        mission_state["persona_id"] = persona_id
    if source_url:
        mission_state["source"] = {"type": "website", "url": source_url}
    session = {
        "id": sid,
        "model": model,
        "agent_key": agent_key if agent_key in AGENT_PROFILES else "sofia",
        "agent_name": agent["name"],
        "agent_role": agent["role"],
        "agent_greeting": agent["greeting"],
        "stage": "chatting",
        "mode": mode,
        "status": "collecting",
        "user_id": user_id,
        "persona_id": persona_id,
        "persona_slug": persona_slug or None,
        "source_url": source_url,
        "initial_block_counts": initial_block_counts,
        "current_block_counts": current_block_counts,
        "knowledge_plan": initial_plan,
        "memory_summary": str(initial_state.get("memory_summary") or "").strip(),
        "plan_changed": False,
        "messages": [],
        "context": _context_with_resume(initial_context or "", resume_meta),
        "mission_state": mission_state,
        "crawler_captures": [],
        "telemetry_transcript": [],
        "telemetry_flags": {"dialog_started_emitted": False},
        "resumed_from_session_id": resume_meta.get("resumed_from_session_id"),
        "resume_source": resume_meta.get("resume_source"),
        "resume_summary": resume_meta.get("resume_summary"),
        "classification": {
            "persona_slug": persona_slug or None,
            "content_type": None,
            "asset_type": None,
            "asset_function": None,
            "title": None,
            "file_ext": None,
            "file_bytes": None,
        },
        "created_at": created_at,
    }
    if initial_plan:
        try:
            initial_plan_state = normalize_validate_summarize_plan(initial_plan, session, live_edit=True)
            _store_plan_state(session, initial_plan_state, last_change="plano inicial normalizado")
        except Exception:
            session["knowledge_plan"] = initial_plan
    # Pre-initialization review: load existing canonical nodes for this persona
    # so the planner and the UI can suggest reuse instead of always creating
    # new audience/product/campaign branches. Best-effort: failures are
    # swallowed so an unavailable Supabase never blocks the session start.
    if mode == "criar" and persona_id:
        try:
            persona_context = _load_persona_context(persona_id)
            session["persona_context"] = persona_context
            session["pre_init_review"] = build_pre_initialization_review(
                session,
                persona_context=persona_context,
                classification=session.get("classification") or {},
            )
        except Exception:
            session.setdefault("persona_context", {ntype: [] for ntype in _PRE_INIT_NODE_TYPES})
            session.setdefault("pre_init_review", {
                "persona_context_loaded": False,
                "existing_nodes_found": {},
                "recommended_connections": [],
                "new_nodes_needed": [],
                "questions": [],
            })
    _sessions[sid] = session
    _save_session(session)
    _emit_kb_event(
        "kb_intake_session_opened",
        session=session,
        source="kb-intake.start",
        status="opened",
        extra={
            "initial_context_present": bool(session.get("context")),
            "initial_context_preview": _truncate(session.get("context"), _EVENT_CONTEXT_PREVIEW_LIMIT),
            "created_at": created_at,
            "resumed_from_session_id": session.get("resumed_from_session_id"),
            "resume_source": session.get("resume_source"),
            "resume_summary": session.get("resume_summary"),
        },
    )
    if mode == "criar":
        _emit_kb_event(
            "kb_intake_preconfirmation_created",
            session=session,
            source="kb-intake.start",
            status="created",
            extra={
                "initial_block_counts": initial_block_counts,
                "current_block_counts": current_block_counts,
                "entry_count": len((initial_plan or {}).get("entries") or []),
                "tree_mode": (initial_plan or {}).get("tree_mode") or "pyramidal",
                "branch_policy": (initial_plan or {}).get("branch_policy") or "top_down_pyramidal",
            },
        )
    return session


def start_bootstrap_session(
    model: str = "gpt-4o-mini",
    initial_context: str = "",
    agent_key: str = "sofia",
    initial_state: Optional[dict[str, Any]] = None,
    bootstrap_llm: bool = True,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if str((initial_state or {}).get("mode") or "").strip().lower() == "criar" and _invalid_criar_persona((initial_state or {}).get("persona_slug")):
        return {
            "ok": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Selecione uma persona especifica antes de criar conhecimento.",
        }
    session = create_session(
        model,
        initial_context=initial_context,
        agent_key=agent_key,
        initial_state=initial_state,
        user_id=user_id,
    )
    if not bootstrap_llm:
        message = _deterministic_bootstrap_message(session)
        session["bootstrap_llm"] = False
        _append_transcript_turn(
            session,
            role="assistant",
            content=message,
            file_attached=False,
            extra={"response_mode": "deterministic_bootstrap"},
        )
        _save_session(session)
        return _bootstrap_result_payload(
            session,
            {
                "message": message,
                "classification": {k: v for k, v in (session.get("classification") or {}).items() if k != "file_bytes"},
                "stage": session.get("stage"),
                "state": session.get("mission_state"),
                "bootstrap_llm": False,
                "timings_ms": {"start": int((time.perf_counter() - started) * 1000)},
            },
        )
    result = chat(session["id"], _BOOTSTRAP_PROMPT, internal=True)
    result.setdefault("timings_ms", {})["start"] = int((time.perf_counter() - started) * 1000)
    if result.get("ok") is False:
        return _bootstrap_result_payload(
            session,
            {
                "message": result.get("message") or "Nao consegui iniciar a conversa automaticamente com o contexto informado.",
                "classification": {k: v for k, v in (session.get("classification") or {}).items() if k != "file_bytes"},
                "stage": session.get("stage"),
                "state": session.get("mission_state"),
                "timings_ms": result.get("timings_ms") or {},
            },
        )
    return _bootstrap_result_payload(session, result)


def get_session(session_id: str) -> Optional[dict]:
    return _get_session(session_id)


def attach_crawler_capture(session_id: str, capture: dict) -> bool:
    session = _get_session(session_id)
    if not session:
        return False
    captures = session.setdefault("crawler_captures", [])
    captures.append(capture)
    session["crawler_captures"] = captures[-5:]
    _save_session(session)
    return True


def attach_reading(session_id: str, reading: dict, file_meta: Optional[dict] = None) -> bool:
    """Append an asset_pipeline reading bundle (compact dict) to the session.

    The reading is also surfaced as a short text line under
    `session['classification']['attachments']` so Sofia/chat downstream can
    pick it up without re-reading the asset.
    """
    session = _get_session(session_id)
    if not session or not isinstance(reading, dict):
        return False
    readings = session.setdefault("asset_readings", [])
    entry = {"file": file_meta or {}, "reading": reading}
    readings.append(entry)
    session["asset_readings"] = readings[-10:]

    extracted = (reading.get("extracted_text") or "").strip()
    visual = (reading.get("visual_summary") or "").strip()
    if extracted or visual:
        classification = session.setdefault("classification", {})
        attachments = classification.setdefault("attachments", [])
        attachments.append({
            "filename": (file_meta or {}).get("filename") or "",
            "kind": reading.get("kind"),
            "extracted_text": extracted[:2000],
            "visual_summary": visual[:400],
            "engine": reading.get("ocr_engine"),
        })
        classification["attachments"] = attachments[-10:]

    _save_session(session)
    return True


def prune_unpersisted_asset_readings(session: dict) -> int:
    """Remove legacy Sofia readings that were never persisted to Storage/assets.

    Older uploads could attach OCR/vision text to the local session even when
    the storage write failed. That made Sofia remember an image that did not
    exist in the assets system. A reading is considered persisted only when it
    carries a public URL or an asset_id/storage path from the upload route.
    """
    if not isinstance(session, dict):
        return 0
    readings = session.get("asset_readings")
    if not isinstance(readings, list) or not readings:
        return 0
    kept: list[dict] = []
    pruned = 0
    for entry in readings:
        file_meta = (entry or {}).get("file") if isinstance(entry, dict) else {}
        if not isinstance(file_meta, dict):
            file_meta = {}
        if file_meta.get("url") or file_meta.get("asset_id") or file_meta.get("storage_path"):
            kept.append(entry)
        else:
            pruned += 1
    if pruned <= 0:
        return 0
    session["asset_readings"] = kept
    cls = session.setdefault("classification", {})
    if not kept and isinstance(cls.get("attachments"), list):
        cls["attachments"] = []
    session["asset_readings_pruned"] = int(session.get("asset_readings_pruned") or 0) + pruned
    _save_session(session)
    return pruned


def update_session_plan(
    session_id: str,
    knowledge_plan: dict[str, Any],
    *,
    status: Optional[str] = None,
    source: str = "kb-intake.session.plan",
    last_change: str = "",
) -> dict[str, Any]:
    session = _get_session(session_id)
    if not session:
        return {"ok": False, "error": "Session not found"}
    if not isinstance(knowledge_plan, dict) or not isinstance(knowledge_plan.get("entries"), list):
        return {"ok": False, "error": "knowledge_plan.entries is required"}
    plan_state = normalize_validate_summarize_plan(knowledge_plan, session, live_edit=True)
    normalized_plan = plan_state["normalized_plan"]
    validation = plan_state["validation"]
    summary = plan_state["summary"]
    counts = summary.get("current_block_counts") or count_blocks_by_type(normalized_plan.get("entries") or [])
    _store_plan_state(session, plan_state, last_change=last_change or "frontend plan sync")
    if status:
        if status == "ready_to_save" and not validation.get("valid"):
            status = "planning"
        session["status"] = status
        if status == "ready_to_save" and validation.get("valid"):
            session["stage"] = "ready_to_save"
            session["confirmed_plan_hash"] = plan_state["plan_hash"]
    elif session.get("stage") == "idle":
        session["stage"] = "chatting"
    _save_session(session)
    event_type = "kb_intake_ready_to_save" if status == "ready_to_save" else "kb_intake_plan_updated"
    _emit_kb_event(
        event_type,
        session=session,
        source=source,
        status=status or "updated",
        extra={
            "current_block_counts": counts,
            "initial_block_counts": session.get("initial_block_counts"),
            "entry_count": summary.get("entry_count"),
            "tree_mode": summary.get("tree_mode"),
            "branch_policy": summary.get("branch_policy"),
            "plan_summary": summary,
            "plan_hash": plan_state.get("plan_hash"),
            "validation": validation,
            "last_change": last_change,
        },
    )
    if "sidebar" in source:
        _emit_kb_event(
            "kb_intake_sidebar_counts_updated",
            session=session,
            source=source,
            status="updated",
            extra={
                "current_block_counts": counts,
                "entry_count": summary.get("entry_count"),
                "plan_hash": plan_state.get("plan_hash"),
            },
        )
    return {
        "ok": True,
        "knowledge_plan": normalized_plan,
        "normalized_plan": normalized_plan,
        "plan_state": plan_state,
        "plan_validation": validation,
        "plan_hash": plan_state.get("plan_hash"),
        "current_block_counts": counts,
        "plan_summary": summary,
        "memory_summary": session.get("memory_summary") or "",
        "status": session.get("status") or session.get("stage"),
        "stage": session.get("stage"),
        "plan_changed": True,
    }


def _source_url_from_context(context: str) -> str | None:
    match = re.search(r"fonte principal:\s*([^\n]+)", context or "", re.I)
    if match:
        coerced = _coerce_urlish_value(match.group(1))
        if coerced:
            return coerced
    match = re.search(r"https?://\S+", context or "")
    if match:
        return match.group(0).strip()
    match = re.search(r"\b(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/[^\s]*)?\b", context or "", re.I)
    return _coerce_urlish_value(match.group(0)) if match else None


def _should_crawl(user_content: str, session: dict) -> bool:
    content = user_content.lower()
    if re.search(r"\b(leia|ler|colete|coletar|site|fonte|link|catalogo|cat[aá]logo)\b", content, re.I):
        return True
    
    # Auto-trigger: URL no contexto + Primeira mensagem da sessao + Sem capturas ainda
    has_url = bool(_source_url_from_context(session.get("context") or ""))
    has_captures = bool(session.get("crawler_captures"))
    is_first_msg = len(session.get("messages", [])) <= 1
    return has_url and not has_captures and is_first_msg


def _crawler_context(captures: list[dict]) -> str:
    if not captures:
        return ""
    latest = captures[-1]
    products = latest.get("product_candidates") or []
    product_lines = []
    for i, product in enumerate(products[:12], 1):
        title = product.get("title") or "sem titulo"
        prices = product.get("prices") or ([product.get("price")] if product.get("price") else [])
        colors = product.get("colors") or []
        product_lines.append(
            f"{i}. {title} | precos={prices or 'pendente'} | cores={colors or 'pendente'} | fonte={product.get('source')}"
        )
    warnings = "\n".join(f"- {w}" for w in latest.get("warnings") or [])
    return "\n".join(
        [
            "Resultado mais recente do crawler heuristico:",
            f"- URL: {latest.get('url')}",
            f"- status_http: {latest.get('status_code')}",
            f"- confianca: {latest.get('confidence')} ({latest.get('confidence_label')})",
            "- politica: captura bruta; validacao humana obrigatoria antes de salvar como conhecimento ativo.",
            "",
            "Candidatos de produtos extraidos:",
            "\n".join(product_lines) if product_lines else "- nenhum candidato confiavel",
            "",
            "Avisos:",
            warnings or "- sem avisos tecnicos",
            "",
            "Preview de texto bruto:",
            (latest.get("raw_text_preview") or "")[:2500],
        ]
    )


# --------------------------------------------------------------------------- #
# Deterministic full-tree builder                                              #
# --------------------------------------------------------------------------- #
# The LLM is unreliable at emitting a complete, valid <knowledge_plan> for a
# big request ("crie toda a arvore"). It frequently stalls ("vou comecar pelo
# briefing"), asks clarifying questions, or emits a plan with broken parent
# links. When the operator EXPLICITLY confirms building the whole tree and we
# already have crawler candidates from the configured source, we build the
# canonical chain server-side and validate it deterministically instead of
# trusting the model JSON. Nothing is hardcoded to a specific client: brand,
# campaign and audience are derived from the persona/source/message, and the
# products come from the real crawler candidates.

# Generic tokens that never identify a product family.
_PRODUCT_FAMILY_STOPWORDS: frozenset[str] = frozenset({
    "oculos", "lupa", "lupas", "kit", "pague", "leve", "para", "com", "sem",
    "de", "do", "da", "novo", "nova", "linha", "modelo", "produto", "produtos",
})


def _full_tree_command(text: str) -> bool:
    """True when the operator confirms building the WHOLE tree.

    Requires a creation/confirmation verb AND a scope word ("toda", "tudo",
    "arvore", "completa", "inspirada"). The first crawl message ("extraia os
    produtos do site ...") deliberately does NOT match so the first turn only
    collects candidates."""
    if not text:
        return False
    low = _fold(text)
    has_verb = bool(re.search(r"\b(cria|crie|criar|monte|monta|montar|gere|gera|gerar|construa|constroi|construir)\b", low))
    confirm = bool(re.match(r"\s*(sim|ok|isso|pode|vai|manda|fecha)\b", low))
    has_scope = bool(re.search(r"\b(toda|todo|tudo|completa|completo|arvore|estrutura|campanha|inspirad[ao])\b", low))
    proceed = bool(re.search(r"\b(gere|gera|gerar|crie|cria|criar)\b.{0,24}\b(prossiga|prosseguir|segue|seguir|continua|continuar)\b", low))
    generate_only = bool(re.match(r"\s*(gere|gera|gerar|crie|cria|criar|monte|monta|montar)\s*[.!?]*\s*$", low))
    extractive_full_scope = bool(
        re.search(r"\b(extraia|extrai|extrair|extracao|captura|capture)\b", low)
        and re.search(r"\b(copys?|copies|copy|campanha|audiencia|audience|publico|publico-alvo|briefing|brand|marca)\b", low)
    )
    extraction_only = bool(re.search(r"\b(extraia|extrai|extrair|extracao|captura|capture|coleta|colete|coletar)\b", low))
    structured_source_spec = bool(
        not extraction_only
        and re.search(r"\bprodutos?\b.{0,32}\b(site|fonte|catalogo|cat[aá]logo)\b|\b(site|fonte|catalogo|cat[aá]logo)\b.{0,32}\bprodutos?\b", low)
        and re.search(r"\b(segmentacao|segmenta[cç][aã]o|publico|p[uú]blico|audiencia|audi[eê]ncia|briefing|campanha|grupo|grupos|agrupad[oa]s?)\b", low)
    )
    return proceed or generate_only or extractive_full_scope or structured_source_spec or (has_scope and (has_verb or confirm))


def _session_user_text(session: dict) -> str:
    # Join with a sentence boundary so per-message phrase extraction (campaign
    # title, audience descriptor) never bleeds across two operator turns.
    return " . ".join(
        str(m.get("content") or "")
        for m in (session.get("messages") or [])
        if m.get("role") == "user"
    )


def _session_latest_candidates(session: dict) -> list[dict]:
    captures = session.get("crawler_captures") or []
    for capture in reversed(captures):
        cands = capture.get("product_candidates") or []
        if cands:
            return cands
    return []


def _session_has_buildable_candidates(session: dict) -> bool:
    return bool(_session_latest_candidates(session))


def _parse_tree_counts(session: dict) -> tuple[int, int]:
    """Parse (num_groups, num_products) from the operator's messages.

    Defaults: 1 group, all available products. Only an explicit numeric
    grouping ("3 grupos") changes the number of groups; bare plural mentions
    stay at one group and are handled as a clarification prompt elsewhere."""
    low = _fold(_session_user_text(session))
    mg = re.search(r"(\d+)\s*grupos?", low) or re.search(
        r"\b(?:agrupo|agrupe|agrupar|agrupamos|agruparia)\s+(\d+)\b",
        low,
    )
    mp = re.search(r"(\d+)\s*produtos?", low)
    num_groups = int(mg.group(1)) if mg else 1
    num_products = int(mp.group(1)) if mp else 0
    num_groups = max(1, num_groups)
    if num_products <= 0:
        num_products = max(num_groups, len(_session_latest_candidates(session)))
    return num_groups, max(num_groups, num_products)


def _requested_single_group_title(session: dict) -> str | None:
    """Return an explicit one-group title from the operator message, if present."""
    text = _session_user_text(session)
    patterns = [
        r"\bgrupos?\s+(?:de\s+produtos?\s+)?([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 _-]{2,40})",
        r"\bgrupo\s+([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 _-]{2,40})",
    ]
    stop = {
        "produto", "produtos", "de", "para", "com", "e", "a", "o", "os", "as",
        "campanha", "fonte", "site", "audiencia", "publico", "padrao",
        "do", "da", "dos", "das",
    }
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if not match:
            continue
        raw = re.split(
            r"[.,;\n]|\b(?:conecte?|campanha|aumentar|fonte|site|extraia|crie|monte)\b",
            match.group(1).strip(),
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        raw = re.sub(r"\s+", " ", raw)
        words = [w for w in raw.split() if _fold(w) not in stop]
        if not words:
            continue
        title = " ".join(words[:3]).strip(" -_")
        if title:
            return title.title()
    return None


def _product_family(title: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ÿ ]+", " ", title or "")
    for token in cleaned.split():
        if len(token) >= 3 and _fold(token) not in _PRODUCT_FAMILY_STOPWORDS:
            return token.strip().title()
    first = (title or "Produto").split()
    return (first[0] if first else "Produto").title()


def _brand_name_for_session(session: dict) -> str:
    slug = session.get("persona_slug") or (session.get("classification") or {}).get("persona_slug") or "marca"
    return slug.replace("-", " ").replace("_", " ").strip().title() or "Marca"


def _brand_slug_base_for_session(session: dict, brand_name: str) -> str:
    base = _slug_for_plan_entry(brand_name or "brand")
    persona_slug = _slug_for_plan_entry(
        session.get("persona_slug")
        or (session.get("classification") or {}).get("persona_slug")
        or ""
    )
    if persona_slug and base == persona_slug:
        return f"{base}-brand"
    return base or "brand"


def _extract_campaign_title(session: dict) -> str:
    text = _session_user_text(session)
    matches = [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in re.finditer(r"(campanha[^.,;\n]{0,80})", text, re.I)
    ]
    if matches:
        def score(raw: str) -> tuple[int, int]:
            folded = _fold(raw)
            generic = folded in {"campanha", "campanha nova", "campanha existente"}
            detail = len([w for w in folded.split() if w not in {"campanha", "nova", "novo", "existente"}])
            return (0 if generic else 1, detail)

        title = max(matches, key=score)
        title = re.sub(r"(?i)^campanha\s+e\s+briefing\s+", "Campanha ", title)
        title = re.sub(r"(?i)^campanha\s+com\s+briefing\s+", "Campanha ", title)
        title = re.sub(r"(?i)^campanha\s*/\s*briefing\s+", "Campanha ", title)
        return title.title()[:80]
    return "Campanha"


def _extract_audience_descriptor(session: dict) -> str:
    text = _session_user_text(session)
    m = re.search(r"\b(?:audiencia|audience|publico(?:-alvo)?)\s+([^.;\n]{3,120})", text, re.I)
    if m:
        base = re.sub(r"\s+", " ", m.group(1)).strip(" :-")
        tail = text[m.end():]
        extra = ""
        next_sentence = re.match(r"\s*[.;,]\s*([^.;\n]{3,100})", tail or "", re.I)
        if next_sentence:
            candidate = re.sub(r"\s+", " ", next_sentence.group(1)).strip(" :-")
            if candidate and not re.match(
                r"(?i)^(campanha|fonte|site|extraia|extrair|crie|criar|monte|montar|produto|produtos|grupo|grupos)\b",
                candidate,
            ):
                extra = candidate
        descriptor = base if not extra else f"{base}: {extra}"
        if descriptor:
            return descriptor[:120].capitalize()
    m = re.search(r"\bloja\s+([^.,;\n]{3,80})", text, re.I)
    if m:
        return ("Publico: " + re.sub(r"\s+", " ", m.group(1)).strip())[:120]
    m = re.search(r"\b(jovens[^.,;\n]{0,80}|publico[^.,;\n]{0,80})", text, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip().capitalize()[:120]
    return "Publico-alvo principal da campanha"


def build_full_tree_plan_from_session(session: dict, user_message: str) -> Optional[dict]:
    """Build a canonical knowledge_plan skeleton from crawler candidates.

    Chain: brand -> briefing -> campaign -> audience -> product_group(s)
    -> product -> copy -> faq. Products are taken from the real crawler
    candidates and distributed across the requested number of groups, grouped
    by detected product family. Returns None when there are no candidates to
    build from. The returned plan is a SKELETON; the caller passes it through
    normalize_validate_summarize_plan for offer/expansion/validation."""
    candidates = _session_latest_candidates(session)
    if not candidates:
        return None

    persona_slug = (
        session.get("persona_slug")
        or (session.get("classification") or {}).get("persona_slug")
        or ""
    )
    num_groups, num_products = _parse_tree_counts(session)
    source_url = session.get("source_url") or ((session.get("mission_state") or {}).get("source") or {}).get("url") or ""

    # Group candidates by detected family, ranked by frequency. When fewer
    # distinct families than requested groups exist, fall back to an even
    # chunking of all candidates so we still honour the grouping request.
    families: dict[str, list[dict]] = {}
    order: list[str] = []
    for cand in candidates:
        fam = _product_family(cand.get("title") or "")
        if fam not in families:
            families[fam] = []
            order.append(fam)
        families[fam].append(cand)
    ranked = sorted(order, key=lambda fam: -len(families[fam]))

    if num_groups == 1:
        group_title = _requested_single_group_title(session) or _product_family(candidates[0].get("title") or "") or "Grupo 1"
        chosen = [(group_title, candidates[:num_products])]
    elif len(ranked) >= num_groups:
        chosen = [(fam, families[fam]) for fam in ranked[:num_groups]]
    else:
        # Even chunking fallback.
        flat = candidates[:num_products]
        size = max(1, -(-len(flat) // num_groups))  # ceil division
        chosen = []
        for i in range(num_groups):
            chunk = flat[i * size:(i + 1) * size]
            if not chunk:
                break
            chosen.append((f"Grupo {i + 1}", chunk))

    per_group = max(1, -(-num_products // max(1, len(chosen))))  # ceil
    entries: list[dict] = []
    links: list[dict] = []
    used: set[str] = set()

    def add(content_type: str, base_slug: str, parent_slug: str, title: str, content: str, metadata: Optional[dict] = None) -> str:
        slug = _dedupe_slug(_slug_for_plan_entry(base_slug)[:60] or content_type, used)
        meta = dict(metadata or {})
        meta["parent_slug"] = parent_slug
        entries.append({
            "content_type": content_type,
            "title": (title or content_type).strip()[:120],
            "slug": slug,
            "status": "pendente_validacao",
            "content": (content or title or content_type).strip(),
            "metadata": meta,
        })
        if parent_slug and parent_slug != "self":
            links.append({"source_slug": parent_slug, "target_slug": slug})
        return slug

    brand_name = _brand_name_for_session(session)
    brand_slug = add(
        "brand",
        _brand_slug_base_for_session(session, brand_name),
        "self",
        brand_name,
        f"Marca {brand_name}, fonte {source_url or 'configurada'}.",
    )
    campaign_title = _extract_campaign_title(session)
    briefing_slug = add(
        "briefing", f"briefing-{campaign_title}", brand_slug,
        f"Briefing {campaign_title}", f"Briefing da {campaign_title}.",
    )
    campaign_slug = add("campaign", campaign_title, briefing_slug, campaign_title, f"{campaign_title}.")
    audience_desc = _extract_audience_descriptor(session)
    audience_slug = add("audience", audience_desc, campaign_slug, audience_desc, audience_desc)

    total = 0
    for fam, items in chosen:
        if total >= num_products:
            break
        group_slug = add("product_group", fam, audience_slug, fam, f"Grupo de produtos {fam}.")
        for cand in items[:per_group]:
            if total >= num_products:
                break
            title = (cand.get("title") or "Produto").strip()
            prices = cand.get("prices") or ([cand.get("price")] if cand.get("price") else [])
            price = prices[0] if prices else None
            meta: dict[str, Any] = {"product_type": fam, "source_url": cand.get("source") or source_url}
            if price:
                meta["price"] = {"unit": {"amount": str(price), "currency": "BRL"}}
            else:
                meta["pending_price"] = True
            product_slug = add(
                "product", title, group_slug, title,
                f"{title}. Fonte: {cand.get('source') or source_url or 'crawler'}.", meta,
            )
            copy_slug = add(
                "copy", f"copy-{product_slug}", product_slug,
                f"Copy {title}", f"Copy de divulgacao para {title}.",
            )
            add(
                "faq", f"faq-{product_slug}", copy_slug,
                f"FAQ {title}", f"Perguntas frequentes sobre {title}.",
            )
            total += 1

    return {
        "source": "session",
        "persona_slug": persona_slug,
        "tree_mode": "pyramidal",
        "entries": entries,
        "links": links,
        "built_by": "deterministic_full_tree",
    }


def _deterministic_tree_summary(plan: dict) -> str:
    counts = count_blocks_by_type((plan or {}).get("entries") or [])
    order = ["brand", "briefing", "campaign", "audience", "product_group", "product", "copy", "faq"]
    parts = [f"{ctype} {counts.get(ctype, 0)}" for ctype in order if counts.get(ctype)]
    return (
        "Cadeia canonica montada: " + " -> ".join(
            p for p in ["brand", "briefing", "campaign", "audience", "product_group", "product", "copy", "faq"]
        )
        + ".\nBlocos criados: " + ", ".join(parts)
        + ".\nAcao: revise a preview e clique em Salvar. Produtos sem preco ficam marcados como pending_price."
    )


# --------------------------------------------------------------------------- #
# Sofia tool-use loop                                                          #
# --------------------------------------------------------------------------- #
# Quando `SOFIA_TOOLS_ENABLED=true` (env var, default false), o chat() invoca
# o LLM passando SOFIA_TOOLS_SCHEMA. Se o modelo responder com tool_calls,
# despachamos cada call para `sofia_tools.dispatch_tool_call`, anexamos o
# resultado como mensagem tool_result e re-prompamos ate o modelo encerrar
# sem tools (ou ate atingir SOFIA_TOOLS_MAX_ITER).
#
# A funcao retorna (raw_text, meta_dict). meta_dict.tool_used=True indica
# que o plano ja foi mutado pelas tools e o caller deve usar o plan_state
# da session (e ignorar `_extract_plan(raw_text)`).
SOFIA_TOOLS_MAX_ITER = 6


def _sofia_tools_enabled() -> bool:
    import os as _os
    return (_os.environ.get("SOFIA_TOOLS_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}


def _invoke_router_with_tools(
    *,
    router: ModelRouter,
    session: dict,
    system_prompt: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Chama o ModelRouter com SOFIA_TOOLS_SCHEMA quando o flag estiver ligado.

    Retorna (raw_text, meta) onde:
      meta["tool_used"] = True quando pelo menos uma tool foi executada.
      meta["tool_calls"] = lista cumulativa das tool_calls executadas.
      meta["provider"] = openai | anthropic (do ultimo turno).
    """
    if not _sofia_tools_enabled():
        result = router.messages_create(
            model=session["model"],
            messages=session["messages"],
            system=system_prompt,
            max_tokens=max_tokens,
        )
        # Quando tools=None, o router devolve string (compatibilidade).
        return (result if isinstance(result, str) else str(result)), {"tool_used": False, "tool_calls": []}

    from services.sofia_tools import SOFIA_TOOLS_SCHEMA, dispatch_tool_call

    # Trabalha em uma copia da lista de mensagens para nao poluir o transcript
    # com tool_result/tool_use raw blocks -- session["messages"] guarda a
    # conversa de operador <-> Sofia em texto.
    working_messages = list(session.get("messages") or [])
    executed_calls: list[dict] = []
    final_text = ""
    last_provider = ""

    for iteration in range(SOFIA_TOOLS_MAX_ITER):
        response = router.messages_create(
            model=session["model"],
            messages=working_messages,
            system=system_prompt,
            max_tokens=max_tokens,
            tools=SOFIA_TOOLS_SCHEMA,
        )
        if isinstance(response, str):
            # Algum branch caiu no caminho legado (ex: provider sem suporte
            # a tools). Devolve direto.
            return response, {"tool_used": bool(executed_calls), "tool_calls": executed_calls, "provider": last_provider}
        if not isinstance(response, dict):
            return str(response), {"tool_used": bool(executed_calls), "tool_calls": executed_calls, "provider": last_provider}
        last_provider = response.get("provider") or last_provider
        text = str(response.get("text") or "")
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            final_text = text
            break
        # Provider-neutral native transcript. ModelRouter converts this to
        # OpenAI assistant.tool_calls + role=tool or Anthropic
        # tool_use/tool_result while preserving the exact call id.
        working_messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("arguments") or {}
            result = dispatch_tool_call(session, name, args if isinstance(args, dict) else {})
            call_id = str(call.get("id") or "").strip()
            if not call_id:
                raise ValueError(f"tool call sem id: {name}")
            executed_calls.append({"id": call_id, "name": name, "arguments": args, "result": result})
            working_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result,
                "is_error": not bool(result.get("ok", False)),
            })
        # Persiste o session a cada iteracao para que mutacoes feitas pelas
        # tools sobrevivam mesmo se a proxima iteracao falhar.
        _save_session(session)
    else:
        # Fim do loop por limite de iteracoes -- pega o texto da ultima resposta.
        final_text = final_text or "Limite de iteracoes do tool-use atingido. Use o plano atual."

    return final_text, {
        "tool_used": bool(executed_calls),
        "tool_calls": executed_calls,
        "provider": last_provider,
    }


def chat(session_id: str, user_message: str, file_info: Optional[dict] = None, internal: bool = False) -> dict:
    """Public chat wrapper that NEVER raises. Any exception escaping the
    real implementation is converted into a controlled `{ok: false, ...}`
    dict so the route layer never has to decide between 500 and 200."""
    try:
        return _chat_impl(session_id, user_message, file_info=file_info, internal=internal)
    except Exception as exc:
        import traceback as _tb
        tb_text = _tb.format_exc()
        try:
            from services import sre_logger
            sre_logger.error(
                "kb_intake_chat_wrapper",
                f"chat() escaped exception session={(session_id or '')[:8]}: {exc}",
                exc,
            )
        except Exception:
            pass
        # Persist the escaped traceback to system_events. The inner handlers
        # (_chat_impl LLM/post-LLM) already persist their failures, but an
        # exception escaping _chat_impl's SETUP phase only reached sre_logger
        # (stdout) — undiagnosable after the fact. This closes that gap so an
        # edit-turn crash leaves a queryable record with the full traceback.
        try:
            from services import supabase_client as _sb
            _sb.insert_event(
                {
                    "event_type": "kb_intake_chat_crashed",
                    "entity_type": "kb_intake_session",
                    "entity_id": session_id,
                    "level": "error",
                    "source": "kb-intake.chat.wrapper",
                    "payload": {
                        "session_id": session_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc)[:500],
                        "user_message_preview": (user_message or "")[:200],
                        "traceback_tail": tb_text.splitlines()[-20:],
                    },
                },
                source="kb-intake.chat.wrapper",
            )
        except Exception:
            pass
        try:
            session = _get_session(session_id)
        except Exception:
            session = None
        return {
            "ok": False,
            "error_code": "INTERNAL_ERROR",
            "exception_type": type(exc).__name__,
            "message": (
                "Nao consegui processar sua mensagem agora. Sua configuracao "
                "foi mantida — tente novamente ou clique em Salvar se ja houver plano."
            ),
            "detail": str(exc)[:300],
            "traceback_tail": tb_text.splitlines()[-12:],
            "state": (session or {}).get("mission_state") if session else None,
        }


def _chat_impl(session_id: str, user_message: str, file_info: Optional[dict] = None, internal: bool = False) -> dict:
    session = _get_session(session_id)
    if not session:
        return {
            "ok": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Sessao nao encontrada.",
            "state": None,
        }

    cls = session["classification"]

    if file_info:
        ext = file_info.get("ext", "")
        cls["file_ext"] = ext
        if file_info.get("bytes"):
            cls["file_bytes"] = file_info["bytes"]
        file_desc = f"[Arquivo: {file_info['filename']} — {len(file_info.get('bytes', b''))} bytes]"
        user_content = f"{file_desc}\n{user_message}".strip() if user_message else file_desc
    else:
        user_content = user_message

    previous_stage = session.get("stage")
    session["messages"].append({"role": "user", "content": user_content})
    _append_transcript_turn(
        session,
        role="user",
        content="Bootstrap automatico com contexto inicial confirmado." if internal else user_content,
        file_attached=bool(file_info),
        extra={
            "file_name": file_info.get("filename") if file_info else None,
            "input_mode": "bootstrap_context" if internal else "user_message",
        },
    )
    mission_state = session.setdefault("mission_state", _default_mission_state(session.get("context") or ""))
    patch = {} if internal else _merge_user_intent(mission_state, user_content)
    progress_reasons: list[str] = []

    # Generation trigger detection — emit kb_intake_generation_requested when
    # the operator types one of the autonomous-generation commands defined in
    # the system prompt (gere/sim/ok/cria/etc). This is the canonical signal
    # for "stop deliberating and produce <knowledge_plan>".
    _GEN_TRIGGER_RE = re.compile(
        r"\b(gere|gera|gerar|cria|criar|crie|construa|monte|monta|montar|"
        r"sim|ok|pode|manda|vai|avanca|avança|continua|continue|"
        r"executa|executar|fecha)\b",
        re.IGNORECASE,
    )
    if not internal and user_content and _GEN_TRIGGER_RE.search(user_content):
        _emit_kb_event(
            "kb_intake_generation_requested",
            session=session,
            source="kb-intake.chat",
            status="requested",
            extra={
                "trigger_message_preview": _truncate(user_content),
                "stage_before": previous_stage,
            },
        )

    flags = session.setdefault("telemetry_flags", {})
    if not flags.get("dialog_started_emitted"):
        progress_reasons.append("dialog_started")
        flags["dialog_started_emitted"] = True
        _emit_kb_event(
            "kb_intake_dialog_started",
            session=session,
            source="kb-intake.chat",
            status="started",
            extra={
                "start_mode": "bootstrap_context" if internal else "user_message",
                "first_user_message_preview": _truncate(user_content),
            },
        )

    crawler_result = None
    if _should_crawl(user_content, session):
        source_url = (mission_state.get("source") or {}).get("url") or _source_url_from_context(session.get("context") or "")
        if source_url:
            try:
                crawler_result = crawl_catalog_url(source_url)
            except Exception as exc:
                crawler_result = {
                    "url": source_url,
                    "confidence": 0.05,
                    "confidence_label": "baixa",
                    "warnings": [f"Crawler indisponivel: {exc}"],
                    "stages": [
                        {"key": "fetch", "label": "captura bruta da URL", "status": "error"},
                        {"key": "validate", "label": "validacao humana obrigatoria", "status": "required"},
                    ],
                }
            attach_crawler_capture(session_id, crawler_result)
            mission_state["evidence_items"] = _build_evidence_items(mission_state, crawler_result)
            progress_reasons.append("crawler_capture")
            if crawler_result.get("warnings"):
                progress_reasons.append("crawler_warning")

    if session.get("mode") == "criar" and session.get("persona_context"):
        try:
            session["pre_init_review"] = build_pre_initialization_review(
                session,
                persona_context=session.get("persona_context") or {},
                classification=session.get("classification") or {},
            )
        except Exception:
            pass

    deterministic_raw: str | None = None
    if (
        not internal
        and _full_tree_command(user_content)
        and _session_has_buildable_candidates(session)
    ):
        built_plan = build_full_tree_plan_from_session(session, user_content)
        if built_plan is not None:
            built_state = normalize_validate_summarize_plan(built_plan, session)
            if built_state["validation"]["valid"]:
                deterministic_plan = built_state["normalized_plan"]
                deterministic_raw = (
                    "Arvore completa criada a partir da fonte "
                    f"{session.get('source_url') or ((mission_state.get('source') or {}).get('url')) or 'configurada'}.\n\n"
                    + _deterministic_tree_summary(deterministic_plan)
                    + "\n\n<knowledge_plan>\n"
                    + json.dumps(deterministic_plan, ensure_ascii=False)
                    + "\n</knowledge_plan>\n"
                    + "<classification>"
                    + json.dumps(
                        {
                            "persona_slug": cls.get("persona_slug") or session.get("persona_slug"),
                            "content_type": "campaign",
                            "title": _extract_campaign_title(session),
                            "complete": True,
                        },
                        ensure_ascii=False,
                    )
                    + "</classification>"
                )

    state_ctx = f"""
Estado atual:
- Agente: {session.get('agent_name') or 'Sofia'} ({session.get('agent_role') or 'agente de inteligencia marketing comercial'})
- Regra de apresentacao: se precisar se apresentar, diga que voce e {session.get('agent_name') or 'Sofia'}; nunca diga que voce e Criar.
- Persona global da sessao: {session.get('persona_slug') or cls['persona_slug'] or '—'}
- Fonte principal: {session.get('source_url') or ((mission_state.get('source') or {}).get('url')) or '—'}
- Tipo de conteúdo: {cls['content_type'] or '—'}
- Tipo de asset: {cls['asset_type'] or '—'}
- Função do asset: {cls['asset_function'] or '—'}
- Título: {cls['title'] or '—'}
- Arquivo binário recebido: {'Sim (' + cls['file_ext'] + ')' if cls.get('file_bytes') else 'Não'}
- Plano inicial: {_format_block_counts(session.get('initial_block_counts'))}
- Plano atual: {_format_block_counts(session.get('current_block_counts'))}
- Memoria viva da sessao: {session.get('memory_summary') or 'plano atual ainda nao expandido'}
- Regra: nao voltar ao plano inicial quando houver knowledge_plan/current_block_counts atualizados.
"""

    if session.get("context"):
        state_ctx += "\nContexto inicial confirmado pelo operador:\n" + session["context"][:6000] + "\n"
    if isinstance(session.get("knowledge_plan"), dict) and (session.get("knowledge_plan") or {}).get("entries"):
        state_ctx += "\nPlano atual vivo em JSON (fonte de verdade para proximas alteracoes):\n"
        state_ctx += json.dumps(session.get("knowledge_plan"), ensure_ascii=False)[:6000] + "\n"
    if session.get("crawler_captures"):
        state_ctx += "\n" + _crawler_context(session["crawler_captures"]) + "\n"
    # Pre-initialization review: surface existing canonical nodes so the
    # planner reuses audience/product/campaign instead of cloning. The agent
    # must ask before duplicating; treat this list as authoritative reuse
    # candidates.
    pre_init_review = session.get("pre_init_review") if isinstance(session.get("pre_init_review"), dict) else None
    if pre_init_review and pre_init_review.get("persona_context_loaded"):
        compact_existing = {
            ntype: [item.get("slug") for item in items if item.get("slug")]
            for ntype, items in (pre_init_review.get("existing_nodes_found") or {}).items()
            if items
        }
        if compact_existing:
            state_ctx += (
                "\nContexto existente da persona (pre-init review): "
                + json.dumps(compact_existing, ensure_ascii=False)[:1500]
                + "\nRegra: antes de propor audience/product/campaign/asset, pergunte ao operador se prefere conectar ao node existente. Nao gere audience nova quando ja houver audience na lista acima.\n"
            )
        recs = pre_init_review.get("recommended_connections") or []
        if recs:
            state_ctx += (
                "Recomendacoes de reuso: "
                + json.dumps(recs[:6], ensure_ascii=False)[:1200]
                + "\n"
            )
        qs = pre_init_review.get("questions") or []
        if qs:
            state_ctx += "Perguntas obrigatorias antes do plano final:\n" + "\n".join(f"- {q}" for q in qs[:4]) + "\n"

    try:
        if deterministic_raw is not None:
            raw = deterministic_raw
            sofia_tools_meta = {"tool_used": False, "tool_calls": [], "provider": "deterministic_full_tree"}
        else:
            router = ModelRouter()
            raw, sofia_tools_meta = _invoke_router_with_tools(
                router=router,
                session=session,
                system_prompt=_SYSTEM_PROMPT + "\n\n" + state_ctx,
                max_tokens=4000,
            )
    except ModelRouterError as exc:
        session["messages"].pop()  # roll back the user message on failure
        _save_session(session)
        _emit_kb_event(
            "kb_intake_dialog_failed",
            session=session,
            source="kb-intake.chat",
            status="failed",
            transcript=True,
            result={
                "error_code": "INTERNAL_ERROR",
                "error_message": f"LLM indisponivel: {exc}",
                "failure_type": "model_router",
            },
        )
        return {
            "ok": False,
            "error_code": "INTERNAL_ERROR",
            "message": f"LLM indisponivel: {exc}",
            "state": mission_state,
        }
    except Exception as exc:
        session["messages"].pop()
        _save_session(session)
        _emit_kb_event(
            "kb_intake_dialog_failed",
            session=session,
            source="kb-intake.chat",
            status="failed",
            transcript=True,
            result={
                "error_code": "INTERNAL_ERROR",
                "error_message": f"Erro inesperado no LLM: {exc}",
                "failure_type": "unexpected",
            },
        )
        return {
            "ok": False,
            "error_code": "INTERNAL_ERROR",
            "message": f"Erro inesperado no LLM: {exc}",
            "state": mission_state,
        }

    try:
        return _process_post_llm_response(
            session=session,
            raw=raw,
            sofia_tools_meta=sofia_tools_meta,
            cls=cls,
            mission_state=mission_state,
            previous_stage=previous_stage,
            user_content=user_content,
            internal=internal,
            crawler_result=crawler_result,
            patch=patch,
            progress_reasons=progress_reasons,
        )
    except Exception as exc:
        import traceback as _tb
        tb_text = _tb.format_exc()
        # 1. Roll back the unpaired user turn (no assistant reply yet) so the
        #    message history and transcript stay paired for the next turn.
        msgs = session.get("messages") or []
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
        transcript = session.get("telemetry_transcript") or []
        if transcript and transcript[-1].get("role") == "user":
            transcript.pop()
            for _i, _item in enumerate(transcript):
                _item["turn_index"] = _i
        # 5. Do not advance save/stage/graph on a post-LLM failure.
        session["stage"] = previous_stage
        # 2. Persist the full traceback to log + telemetry event.
        try:
            from services import sre_logger
            sre_logger.error(
                "kb_intake_chat_post_llm",
                f"post-LLM processing failed session={(session_id or '')[:8]}: {exc}",
                exc,
            )
        except Exception:
            pass
        try:
            _emit_kb_event(
                "kb_intake_dialog_failed",
                session=session,
                source="kb-intake.chat",
                status="failed",
                transcript=True,
                result={
                    "error_code": "POST_LLM_ERROR",
                    "error_message": str(exc)[:500],
                    "failure_type": "post_llm",
                    "traceback_tail": tb_text.splitlines()[-12:],
                },
            )
        except Exception:
            pass
        _save_session(session)
        # 3. Useful response carrying the technical reason + traceback tail.
        return {
            "ok": False,
            "error_code": "POST_LLM_ERROR",
            "exception_type": type(exc).__name__,
            "message": (
                "Recebi a resposta do modelo, mas falhei ao processar o plano. "
                "Revertendo sua ultima mensagem para voce tentar de novo sem "
                "duplicar o historico. Detalhe tecnico em traceback_tail."
            ),
            "detail": str(exc)[:300],
            "traceback_tail": tb_text.splitlines()[-12:],
            "stage": session.get("stage"),
            "state": mission_state,
        }


def _process_post_llm_response(
    *,
    session: dict,
    raw: str,
    sofia_tools_meta: dict,
    cls: dict,
    mission_state: dict,
    previous_stage,
    user_content,
    internal: bool,
    crawler_result,
    patch: dict,
    progress_reasons: list,
) -> dict:
    """Deterministic post-LLM processing extracted from _chat_impl.

    Plan extraction, normalization, visible summary and telemetry. Isolated in
    its own function so the entire post-LLM phase runs inside one resilience
    boundary: any failure is caught by the caller, which rolls back the unpaired
    user turn and returns a controlled error instead of corrupting the
    transcript or advancing save/graph."""
    cls_data = _extract_cls(raw)
    plan_payload = _extract_plan(raw)
    plan_changed = False
    current_block_counts = _normalize_block_counts(session.get("current_block_counts"))
    plan_violations: list[str] = []
    plan_summary: dict[str, Any] | None = None
    plan_state: dict[str, Any] | None = None

    # Quando o Sofia tool-use loop ja mutou o plano via sofia_tools, o
    # session["normalized_plan"] e o plan_state mais frescos estao em
    # session["plan_validation"] / session["plan_summary"]. Nesses casos o
    # LLM raramente repete <knowledge_plan> -- consumimos direto da session.
    if sofia_tools_meta.get("tool_used") and not plan_payload:
        normalized_from_session = session.get("normalized_plan") or {}
        if isinstance(normalized_from_session, dict) and normalized_from_session.get("entries"):
            plan_state = {
                "normalized_plan": normalized_from_session,
                "validation": session.get("plan_validation") or {"valid": True, "blocking_violations": [], "warnings": []},
                "summary": session.get("plan_summary") or summarize_normalized_plan(normalized_from_session),
                "plan_hash": str(session.get("plan_hash") or _plan_hash(normalized_from_session)),
            }
            plan_payload = plan_state["normalized_plan"]
            plan_violations = plan_state["validation"].get("blocking_violations") or []
            plan_summary = plan_state["summary"]
            current_block_counts = plan_summary.get("current_block_counts") or current_block_counts
            plan_changed = not plan_violations
            session["status"] = "planning" if plan_violations else session.get("status") or "planning"
    if plan_payload and plan_state is None:
        plan_state = normalize_validate_summarize_plan(plan_payload, session)
        plan_payload = plan_state["normalized_plan"]
        plan_violations = plan_state["validation"]["blocking_violations"]
        plan_summary = plan_state["summary"]
        _store_plan_state(session, plan_state, last_change="knowledge_plan gerado pela Sofia")
        if plan_violations:
            session["plan_validation_warnings"] = plan_violations
            session["last_invalid_plan"] = plan_payload
            current_block_counts = plan_summary["current_block_counts"]
            session["status"] = "planning"
        else:
            session.pop("plan_validation_warnings", None)
            current_block_counts = plan_summary["current_block_counts"]
            plan_changed = True

    # Deterministic full-tree materialization. When the operator explicitly
    # confirms building the entire tree and the LLM produced no usable plan (or
    # one with blocking violations), build the canonical chain from the crawler
    # candidates so the Create path materializes instead of stalling. Never
    # depends on the LLM emitting a perfect giant JSON.
    llm_plan_usable = bool(isinstance(plan_payload, dict) and plan_payload.get("entries")) and not plan_violations
    if (
        not internal
        and _full_tree_command(user_content)
        and not llm_plan_usable
        and _session_has_buildable_candidates(session)
    ):
        built_plan = build_full_tree_plan_from_session(session, user_content)
        if built_plan is not None:
            built_state = normalize_validate_summarize_plan(built_plan, session)
            if built_state["validation"]["valid"]:
                plan_state = built_state
                plan_payload = built_state["normalized_plan"]
                plan_violations = []
                plan_summary = built_state["summary"]
                _store_plan_state(session, built_state, last_change="arvore completa construida a partir da fonte")
                current_block_counts = plan_summary["current_block_counts"]
                plan_changed = True
                session.pop("plan_validation_warnings", None)
                session.pop("last_invalid_plan", None)
                session["status"] = "ready_to_save"
                session["stage"] = "ready_to_save"
                cls["persona_slug"] = cls.get("persona_slug") or session.get("persona_slug")
                cls["content_type"] = cls.get("content_type") or "campaign"
                cls["title"] = cls.get("title") or _extract_campaign_title(session)
                raw = (
                    "Arvore completa criada a partir da fonte "
                    f"{session.get('source_url') or 'configurada'}.\n\n"
                    + _deterministic_tree_summary(plan_payload)
                )

    visible = _strip_knowledge_plan(_strip_cls(raw))
    if plan_violations:
        visible = (
            f"{visible}\n\nPlano recebido, mas bloqueado antes da preview/save por violacoes:\n- "
            + "\n- ".join(plan_violations)
        ).strip()
    visible = _rewrite_visible_plan_summary(visible, plan_payload if isinstance(plan_payload, dict) else None)
    plan_entries = plan_payload.get("entries", []) if isinstance(plan_payload, dict) and not plan_violations else []

    if cls_data:
        for key in ("persona_slug", "content_type", "asset_type", "asset_function", "title"):
            if cls_data.get(key):
                cls[key] = cls_data[key]
        if cls_data.get("complete") and not plan_violations:
            session["stage"] = "ready_to_save"
    if plan_violations:
        session["stage"] = "chatting"
        session["status"] = "planning"

    session["messages"].append({"role": "assistant", "content": raw})
    _append_transcript_turn(
        session,
        role="assistant",
        content=visible,
        extra={
            "has_knowledge_plan": bool(plan_entries),
            "response_mode": "bootstrap_context" if internal else "user_message",
        },
    )
    _apply_save_inference(session)

    # Auto-promote to ready_to_save when a valid plan is emitted and the
    # required classification fields are present. The model frequently emits
    # the <knowledge_plan> block but forgets to flip <classification> "complete"
    # to true, which leaves the operator without a Save button. The plan
    # itself is the strongest signal that the agent considers itself done.
    auto_promoted = False
    if (
        session.get("stage") != "ready_to_save"
        and plan_entries
        and cls.get("persona_slug")
        and cls.get("content_type")
        and cls.get("title")
    ):
        session["stage"] = "ready_to_save"
        auto_promoted = True

    if plan_entries:
        try:
            entry_types: list[str] = []
            for e in plan_entries:
                t = e.get("content_type") if isinstance(e, dict) else None
                if t:
                    entry_types.append(t)
            _emit_kb_event(
                "kb_intake_generation_completed",
                session=session,
                source="kb-intake.chat",
                status="generated",
                extra={
                    "entry_count": len(plan_entries),
                    "entry_types": entry_types,
                    "auto_promoted_stage": auto_promoted,
                    "plan_summary": plan_summary,
                },
            )
        except Exception:
            pass

        try:
            _emit_kb_event(
                "kb_intake_plan_updated",
                session=session,
                source="kb-intake.chat",
                status="updated",
                extra={
                    "current_block_counts": current_block_counts,
                    "initial_block_counts": session.get("initial_block_counts"),
                    "entry_count": len(plan_entries),
                    "tree_mode": plan_payload.get("tree_mode") if isinstance(plan_payload, dict) else None,
                    "branch_policy": plan_payload.get("branch_policy") if isinstance(plan_payload, dict) else None,
                    "plan_summary": plan_summary,
                },
            )
        except Exception:
            pass

    if session.get("stage") == "ready_to_save" and previous_stage != "ready_to_save" and not plan_violations:
        session["status"] = "ready_to_save"
        try:
            _emit_kb_event(
                "kb_intake_ready_to_save",
                session=session,
                source="kb-intake.chat",
                status="ready_to_save",
                extra={
                    "entry_count": len(plan_entries) if plan_entries else 0,
                    "from_stage": previous_stage,
                    "auto_promoted": auto_promoted,
                },
            )
        except Exception:
            pass

    if patch:
        progress_reasons.append("intent_patch")
    if plan_entries:
        progress_reasons.append("knowledge_plan_generated")
    if previous_stage != session.get("stage"):
        progress_reasons.append("stage_changed")

    missing_targets: list[str] = []
    for model in mission_state.get("requested_outputs", {}).get("models", []):
        req = int(model.get("products_requested") or 0)
        if req <= 0:
            continue
        found = len([
            e for e in mission_state.get("evidence_items") or []
            if _fold(str(e.get("model") or "")) == _fold(str(model.get("name") or ""))
        ])
        if found < req:
            missing_targets.append(f"{model.get('name')}: {found}/{req}")
    if missing_targets:
        mission_state["status"] = "partial_collection"
    elif mission_state.get("evidence_items"):
        mission_state["status"] = "collected"

    prefix = _mission_summary(mission_state) if patch else ""
    if missing_targets:
        prefix += (
            ("\n\n" if prefix else "")
            + "Coleta parcial: " + ", ".join(missing_targets)
            + ". Posso complementar manualmente ou buscar outra fonte."
        )
    visible_out = f"{prefix}\n\n{visible}".strip() if prefix else visible

    progress_reasons = [reason for reason in progress_reasons if reason != "dialog_started"]
    if internal:
        progress_reasons.append("bootstrap_response_generated")
    if progress_reasons:
        _emit_kb_event(
            "kb_intake_dialog_progress",
            session=session,
            source="kb-intake.chat",
            status="in_progress",
            extra={
                "progress_reasons": progress_reasons,
                "patch_keys": sorted((patch or {}).keys()),
                "crawler_confidence": (crawler_result or {}).get("confidence") if crawler_result else None,
                "missing_targets": missing_targets,
                "input_mode": "bootstrap_context" if internal else "user_message",
            },
        )

    _save_session(session)

    return {
        "ok": True,
        "message": visible_out,
        "stage": session["stage"],
        "classification": {k: v for k, v in cls.items() if k != "file_bytes"},
        "crawler": crawler_result,
        "proposed_entries": plan_entries,
        "proposed_plan": plan_payload or None,
        "knowledge_plan": plan_payload or session.get("knowledge_plan"),
        "current_block_counts": current_block_counts,
        "memory_summary": session.get("memory_summary") or "",
            "plan_summary": plan_summary,
            "plan_violations": plan_violations,
            "plan_state": plan_state,
            "normalized_plan": plan_payload,
            "plan_validation": (plan_state or {}).get("validation") if plan_state else None,
            "plan_hash": (plan_state or {}).get("plan_hash") if plan_state else None,
            "plan_changed": plan_changed,
        "state": mission_state,
        "patch": patch,
    }


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\s\-]", "_", name).strip().replace(" ", "_")


def _fold(text: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFKD", text or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def _infer_from_transcript(session: dict) -> dict:
    transcript = "\n".join(str(m.get("content") or "") for m in session.get("messages", []))
    visible = _strip_cls(transcript)
    folded = _fold(visible)
    inferred: dict = {}

    def label_key(label: str) -> str | None:
        normalized = _fold(label).strip().replace("?", "")
        if normalized == "cliente":
            return "persona_slug"
        if normalized in {"tipo", "tipo de conteudo", "tipo de contedo"}:
            return "content_type"
        if normalized in {"titulo", "ttulo"}:
            return "title"
        if normalized in {"descricao", "descrio"}:
            return "description"
        if normalized == "link":
            return "link"
        return None
    for line in visible.splitlines():
        clean = line.strip().lstrip("-").strip()
        if ":" not in clean:
            continue
        label, value = clean.split(":", 1)
        key = label_key(label)
        if key and value.strip():
            inferred[key] = value.strip().strip("-").strip()
        # Otimizacao: ignora linhas muito longas ou se ja encontrou os metadados principais
        if len(line) > 200 or len(inferred) >= 5:
            break

    if inferred.get("persona_slug"):
        key = _fold(inferred["persona_slug"]).strip()
        inferred["persona_slug"] = _PERSONA_ALIASES.get(key, key.replace(" ", "-"))
    else:
        for alias, slug in _PERSONA_ALIASES.items():
            if alias in folded:
                inferred["persona_slug"] = slug
                break

    if inferred.get("content_type"):
        key = _fold(inferred["content_type"]).strip()
        inferred["content_type"] = _CONTENT_ALIASES.get(key, key)
    else:
        for alias, ctype in _CONTENT_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", folded):
                inferred["content_type"] = ctype
                break

    return inferred


def _apply_save_inference(session: dict) -> None:
    cls = session["classification"]
    inferred = _infer_from_transcript(session)
    
    # 1. Inferir campos básicos do texto do chat
    for key in ("persona_slug", "content_type", "title"):
        if inferred.get(key) and (not cls.get(key) or cls.get(key) == "other"):
            cls[key] = inferred[key]
            
    # 2. Fallback: Se o título ainda estiver faltando, buscar no plano de conhecimento
    if not cls.get("title"):
        for msg in reversed(session.get("messages", [])):
            if msg.get("role") == "assistant":
                entries = _extract_plan_entries(msg.get("content") or "")
                if entries:
                    # Pega o título da primeira entrada ou do briefing
                    briefing = next((e for e in entries if e.get("content_type") == "briefing"), entries[0])
                    cls["title"] = briefing.get("title")
                    break

    # 3. Último recurso: Inferir da URL ou Persona para não travar o salvamento
    if not cls.get("title"):
        url = _source_url_from_context(session.get("context") or "")
        if url:
            cls["title"] = f"Extracao: {url.split('//')[-1]}"
        elif cls.get("persona_slug"):
            cls["title"] = f"Conhecimento: {cls['persona_slug']}"

    if inferred.get("description"):
        cls["description"] = inferred["description"]
    if inferred.get("link"):
        cls["link"] = inferred["link"]

    # If a previous assistant turn already produced a <knowledge_plan> and the
    # classification has the required fields, promote the session even if the
    # model forgot to mark complete:true. Without this the operator never sees
    # the Save button while the model keeps hallucinating "salvo com sucesso".
    if session.get("stage") != "ready_to_save" and cls.get("persona_slug") and cls.get("content_type") and cls.get("title"):
        for msg in session.get("messages", []):
            if msg.get("role") == "assistant" and "<knowledge_plan>" in (msg.get("content") or ""):
                if _extract_plan_entries(msg.get("content") or ""):
                    session["stage"] = "ready_to_save"
                    break


def _build_content(session: dict, content_text: str) -> str:
    if content_text and content_text.strip():
        return content_text.strip()
    cls = session["classification"]
    inferred = _infer_from_transcript(session)
    description = cls.get("description") or inferred.get("description") or ""
    link = cls.get("link") or inferred.get("link") or ""

    if cls.get("content_type") == "faq":
        lines = [f"Pergunta: {cls.get('title') or 'FAQ'}"]
        lines.append(f"Resposta: {description}" if description else "Resposta: ")
        if link:
            lines.extend(["", f"Link: {link}"])
        return "\n".join(lines)

    lines: list[str] = []
    if description:
        lines.extend(["## Descrição", "", description, ""])
    if link:
        lines.extend(["## Link", "", link, ""])
    return "\n".join(lines).strip()


def _vault_client_folder(persona_slug: Optional[str]) -> str:
    if not persona_slug:
        return _GLOBAL_VAULT_CLIENT_FOLDER
    return persona_folder_name(persona_slug)


def _slug_for_plan_entry(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    return text.strip("-") or "item"


def _fallback_plan_content(session: dict, content_text: str) -> str:
    built = _build_content(session, content_text)
    if built.strip():
        return built.strip()
    recent_messages: list[str] = []
    for msg in reversed(session.get("messages", [])):
        raw = _strip_cls(str(msg.get("content") or "")).strip()
        raw = re.sub(r"<knowledge_plan>.*?</knowledge_plan>", "", raw, flags=re.DOTALL).strip()
        if not raw:
            continue
        recent_messages.append(f"{msg.get('role')}: {raw}")
        if len(recent_messages) >= 4:
            break
    if recent_messages:
        return "## Transcript\n\n" + "\n\n".join(reversed(recent_messages))
    cls = session.get("classification") or {}
    return f"Conhecimento capturado para {cls.get('title') or 'item sem titulo'}."


def _fallback_plan_payload(session: dict, content_text: str) -> dict:
    cls = session.get("classification") or {}
    inferred = _infer_from_transcript(session)
    entry_slug = _slug_for_plan_entry(cls.get("title") or inferred.get("title") or "item")
    metadata = {
        **(cls.get("metadata") or {}),
        "slug": entry_slug,
        "parent_slug": "self",
        "generated_from": "session_fallback",
        "link": cls.get("link") or inferred.get("link"),
        "asset_type": cls.get("asset_type"),
        "asset_function": cls.get("asset_function"),
    }
    metadata = {k: v for k, v in metadata.items() if v not in (None, "", [], {})}
    return {
        "source": cls.get("link") or inferred.get("link") or "session_fallback",
        "persona_slug": cls.get("persona_slug"),
        "validation_policy": "human_validation_required",
        "entries": [
            {
                "content_type": cls.get("content_type") or "other",
                "title": cls.get("title") or inferred.get("title") or "Conhecimento",
                "slug": entry_slug,
                "status": "pendente_validacao",
                "content": _fallback_plan_content(session, content_text),
                "tags": cls.get("tags") or inferred.get("tags") or [],
                "metadata": metadata,
            }
        ],
        "links": [],
        "missing_questions": [],
    }


def _write_entry_file(persona_slug: str, entry: dict) -> Optional[Path]:
    """Salva uma entrada individual de um plano de conhecimento no vault."""
    vault_root = Path(VAULT_PATH)
    ensure_persona_vault_structure(persona_slug, VAULT_PATH)
    client_folder = _vault_client_folder(persona_slug)
    content_type = entry.get("content_type") or "other"
    type_folder = _CONTENT_TYPE_FOLDERS.get(content_type, "00_OTHER")
    
    # Prefere o slug como base do nome do arquivo, fallback para o titulo
    base_name = entry.get("slug") or entry.get("title") or "untitled"
    safe_name = _safe_filename(base_name)

    target_dir = vault_root / "AI-BRAIN" / "05_ENTITIES" / "CLIENTS" / client_folder / type_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{safe_name}.md"
    target_path = target_dir / filename

    now = datetime.now(timezone.utc).isoformat()
    # Monta frontmatter rico
    fm = {
        "title": entry.get("title"),
        "client": persona_slug,
        "type": content_type,
        "slug": entry.get("slug"),
        "created_at": now,
        "status": entry.get("status", "pendente_validacao"),
        "created_via": "kb_intake_sofia",
        "sync_origin": "direct_save",
    }
    if entry.get("metadata"):
        fm.update(entry["metadata"])
    if entry.get("tags"):
        fm["tags"] = entry["tags"]

    fm_lines = ["---"]
    for k, v in fm.items():
        val = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        fm_lines.append(f"{k}: {val}")
    fm_lines.append("---")

    body = entry.get("content") or ""
    target_path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")
    return target_path


def _write_file(session: dict, content_text: str) -> Path:
    cls = session["classification"]
    vault_root = Path(VAULT_PATH)
    ensure_persona_vault_structure(cls["persona_slug"], VAULT_PATH)
    client_folder = _vault_client_folder(cls["persona_slug"])
    type_folder = _CONTENT_TYPE_FOLDERS.get(cls["content_type"] or "other", "00_OTHER")
    safe_title = _safe_filename(cls["title"] or "untitled")

    ext = cls.get("file_ext") or ""
    is_binary_asset = ext.lower() in _ASSET_EXTS and cls.get("file_bytes")

    if is_binary_asset:
        target_dir = vault_root / "AI-BRAIN" / "05_ENTITIES" / "CLIENTS" / client_folder / "assets"
        filename = f"{safe_title}{ext}"
    else:
        target_dir = vault_root / "AI-BRAIN" / "05_ENTITIES" / "CLIENTS" / client_folder / type_folder
        filename = f"{safe_title}.md"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if is_binary_asset:
        target_path.write_bytes(cls["file_bytes"])
    else:
        now = datetime.now(timezone.utc).isoformat()
        lines = ["---", f"title: {cls['title']}", f"client: {cls['persona_slug']}",
                 f"type: {cls['content_type']}", "created_via: kb_intake_sofia", "sync_origin: direct_save"]
        if cls.get("link"):
            lines.append(f"link: {cls['link']}")
        if cls.get("asset_type"):
            lines.append(f"asset_type: {cls['asset_type']}")
        if cls.get("asset_function"):
            lines.append(f"asset_function: {cls['asset_function']}")
        lines += [f"created_at: {now}", "---", "", content_text or ""]
        target_path.write_text("\n".join(lines), encoding="utf-8")

    return target_path


def _git_ops(vault_path: str, rel_path: str, title: str, client: str) -> dict:
    def run(args: list, **kw) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=vault_path, capture_output=True, text=True, timeout=60, **kw)

    add = run(["git", "add", rel_path])
    commit = run(["git", "commit", "-m", f"kb: add {title} [{client}]"])
    push = run(["git", "push"])

    return {
        "add_ok": add.returncode == 0,
        "commit_ok": commit.returncode == 0,
        "push_ok": push.returncode == 0,
        "commit_out": commit.stdout.strip()[:200],
        "push_err": push.stderr.strip()[:200],
    }


def save(session_id: str, content_text: str = "", plan_override: Optional[dict] = None) -> dict:
    session = _get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    try:
        _emit_kb_event(
            "kb_intake_save_requested",
            session=session,
            source="kb-intake.save",
            status="requested",
            extra={
                "stage": session.get("stage"),
                "has_content_text_override": bool(content_text and content_text.strip()),
                "has_plan_override": bool(isinstance(plan_override, dict) and (plan_override.get("entries") or plan_override.get("normalized_plan") or plan_override.get("plan_hash"))),
            },
        )
    except Exception:
        pass

    _apply_save_inference(session)
    cls = session["classification"]
    missing = [k for k in ("persona_slug", "content_type", "title") if not cls.get(k)]
    if missing:
        _emit_kb_event(
            "kb_intake_dialog_rejected",
            session=session,
            source="kb-intake.save",
            status="rejected",
            transcript=True,
            result={
                "error": "Classification incomplete",
                "missing_fields": missing,
                "classification": {k: v for k, v in cls.items() if k != "file_bytes"},
            },
        )
        return {
            "error": "Classification incomplete - missing " + ", ".join(missing),
            "classification": {k: v for k, v in cls.items() if k != "file_bytes"},
        }

    session_plan_hash = str(session.get("plan_hash") or "")
    plan_payload: dict = {}
    plan_state: dict[str, Any] | None = None
    if isinstance(plan_override, dict):
        override_hash = str(plan_override.get("plan_hash") or "")
        override_plan = plan_override.get("normalized_plan") if isinstance(plan_override.get("normalized_plan"), dict) else None
        if override_plan is None and isinstance(plan_override.get("entries"), list):
            override_plan = plan_override
        if override_hash and session_plan_hash and override_hash != session_plan_hash:
            error = "Plan mismatch: save payload is not the current normalized plan."
            _emit_kb_event(
                "kb_intake_dialog_rejected",
                session=session,
                source="kb-intake.save",
                status="rejected",
                transcript=True,
                result={"error": error, "session_plan_hash": session_plan_hash, "save_plan_hash": override_hash},
            )
            return {"error": error, "session_plan_hash": session_plan_hash, "save_plan_hash": override_hash}
        if isinstance(override_plan, dict) and override_plan.get("entries"):
            if override_hash and override_hash == session_plan_hash and isinstance(session.get("normalized_plan"), dict):
                plan_payload = dict(session.get("normalized_plan") or {})
                plan_state = _session_public_state(session).get("plan_state")
            else:
                plan_state = normalize_validate_summarize_plan(override_plan, session, live_edit=True)
                if session_plan_hash and plan_state["plan_hash"] != session_plan_hash:
                    error = "Plan mismatch: save payload is not the current normalized plan."
                    _emit_kb_event(
                        "kb_intake_dialog_rejected",
                        session=session,
                        source="kb-intake.save",
                        status="rejected",
                        transcript=True,
                        result={"error": error, "session_plan_hash": session_plan_hash, "save_plan_hash": plan_state["plan_hash"]},
                    )
                    return {"error": error, "session_plan_hash": session_plan_hash, "save_plan_hash": plan_state["plan_hash"]}
                plan_payload = plan_state["normalized_plan"]

    if not plan_payload and isinstance(session.get("normalized_plan"), dict) and (session.get("normalized_plan") or {}).get("entries"):
        plan_payload = dict(session.get("normalized_plan") or {})
        plan_state = _session_public_state(session).get("plan_state")
    elif not plan_payload and isinstance(session.get("knowledge_plan"), dict) and (session.get("knowledge_plan") or {}).get("entries"):
        plan_state = normalize_validate_summarize_plan(dict(session.get("knowledge_plan") or {}), session, live_edit=True)
        plan_payload = plan_state["normalized_plan"]
        _store_plan_state(session, plan_state, last_change="save normalized legacy plan")
    elif not plan_payload:
        plan_state = normalize_validate_summarize_plan(_fallback_plan_payload(session, content_text), session)
        plan_payload = plan_state["normalized_plan"]
        _store_plan_state(session, plan_state, last_change="save fallback plan")

    if plan_state is None:
        plan_state = _plan_state_from_normalized(plan_payload, session=session)
    validation = plan_state.get("validation") or _plan_validation()
    if not validation.get("valid"):
        error = "Plano ainda não pode ser salvo. Corrija as pendências bloqueantes primeiro."
        _emit_kb_event(
            "kb_intake_dialog_rejected",
            session=session,
            source="kb-intake.save",
            status="rejected",
            transcript=True,
            result={"error": error, "violations": validation.get("blocking_violations") or []},
        )
        return {"error": error, "violations": validation.get("blocking_violations") or [], "plan_state": plan_state}

    if str(session.get("mode") or "").strip().lower() == "criar":
        current_hash = str(plan_state.get("plan_hash") or "")
        confirmed_hash = str(session.get("confirmed_plan_hash") or "")
        if current_hash and confirmed_hash != current_hash:
            error = "Confirme a estrutura no chat antes de salvar. O JSON atual ainda não foi confirmado pelo operador."
            _emit_kb_event(
                "kb_intake_dialog_rejected",
                session=session,
                source="kb-intake.save",
                status="rejected",
                transcript=True,
                result={
                    "error": error,
                    "error_code": "PLAN_CONFIRMATION_REQUIRED",
                    "plan_hash": current_hash,
                    "confirmed_plan_hash": confirmed_hash,
                },
            )
            return {
                "error": error,
                "error_code": "PLAN_CONFIRMATION_REQUIRED",
                "plan_hash": current_hash,
                "confirmed_plan_hash": confirmed_hash,
                "plan_state": plan_state,
            }

    plan_entries = plan_payload.get("entries", [])
    expected_counts = _normalize_block_counts((plan_state.get("summary") or {}).get("current_block_counts") or session.get("current_block_counts"))
    actual_counts = count_blocks_by_type(plan_entries)
    mismatch = _count_mismatch_message(expected_counts, actual_counts)
    if mismatch or len(plan_entries) != int((plan_state.get("summary") or {}).get("entry_count") or len(plan_entries)):
        mismatch = mismatch or "Plan mismatch: normalized plan entry_count differs from save payload entry_count."
        _emit_kb_event(
            "kb_intake_dialog_rejected",
            session=session,
            source="kb-intake.save",
            status="rejected",
            transcript=True,
            result={
                "error": mismatch,
                "current_block_counts": expected_counts,
                "save_payload_counts": actual_counts,
            },
        )
        return {
            "error": mismatch,
            "current_block_counts": expected_counts,
            "save_payload_counts": actual_counts,
        }
    _emit_kb_event(
        "kb_intake_save_payload_validated",
        session=session,
        source="kb-intake.save",
        status="validated",
        extra={
            "current_block_counts": expected_counts,
            "save_payload_counts": actual_counts,
            "entry_count": len(plan_entries),
            "tree_mode": (plan_state.get("summary") or {}).get("tree_mode") or "pyramidal",
            "branch_policy": (plan_state.get("summary") or {}).get("branch_policy") or "top_down_pyramidal",
            "plan_summary": plan_state.get("summary"),
            "plan_hash": plan_state.get("plan_hash"),
        },
    )
    plan_warnings = [
        warning
        for warning in (plan_payload.get("warnings") or [])
        if isinstance(warning, dict)
    ]

    if str(session.get("mode") or "").strip().lower() == "criar":
        if _persona_uses_graph_bundle_pipeline(str(session.get("persona_id") or "")):
            return _save_via_graph_bundle(
                session, session_id, plan_payload, plan_state, plan_warnings,
            )
        try:
            graph_doc = GraphJson.model_validate(plan_state.get("graph_json") or normalized_plan_to_graph_json(plan_payload, session).model_dump())
            publication = graph_document_publisher.publish(
                graph=graph_doc,
                persona_slug=graph_doc.persona_slug,
                brand_slug=graph_doc.brand_slug,
                source="kb-intake.save",
                session_id=session_id,
                idempotency_key=f"kb-intake.save:{session_id}:{plan_state.get('plan_hash') or 'current'}",
            )
            import_result = {**publication, **(publication.get("projections") or {})}
        except Exception as exc:
            import_result = {
                "ok": False,
                "error_code": "GRAPH_JSON_IMPORT_FAILED",
                "errors": [str(exc)],
            }
        if import_result.get("ok") is False:
            error_code = import_result.get("error_code") or "GRAPH_JSON_INVALID"
            errors = import_result.get("errors") or [import_result.get("error") or "Graph JSON import failed"]
            response = {
                "error": "O grafo JSON ainda nao pode ser salvo. A Sofia precisa corrigir a estrutura antes da importacao.",
                "error_code": error_code,
                "requires_sofia_intervention": True,
                "graph_json_validation": {
                    "blocking": errors,
                    "questions": [
                        "Qual deve ser o pai principal dos nodes sem caminho ate Persona?",
                        "A FAQ deve ficar abaixo de Copy, Product ou Product Group neste galho?",
                    ],
                },
                "plan_state": plan_state,
            }
            _emit_kb_event(
                "kb_intake_dialog_rejected",
                session=session,
                source="kb-intake.save",
                status="rejected",
                transcript=True,
                result=response,
            )
            return response

        session["stage"] = "done"
        session["status"] = "saved"
        _save_session(session)
        completion_payload = {
            "file_path": (import_result.get("written_files") or ["graph-json-import"])[0],
            "saved_paths": import_result.get("written_files") or [],
            "git": {"ok": True, "git": "skipped_graph_json_import", "commit_ok": True, "push_ok": True},
            "success": True,
            "status": "saved",
            "warnings": plan_warnings,
            "sync_mode": "graph_json_import",
            "entries_written": len(import_result.get("written_files") or []),
            "plan_entries": len(plan_entries),
            "plan_links": len(plan_payload.get("links") or []),
            "knowledge_item_ids": import_result.get("knowledge_item_ids") or [],
            "knowledge_node_ids": import_result.get("knowledge_node_ids") or [],
            "knowledge_edge_ids": import_result.get("knowledge_edge_ids") or [],
            "hierarchy": {
                "mode": "graph_json_import",
                "nodes_imported": import_result.get("nodes_imported"),
                "edges_imported": import_result.get("edges_imported"),
            },
            "plan_state": plan_state,
            "plan_hash": plan_state.get("plan_hash"),
            "graph_json": graph_doc.model_dump(),
            "vault_write": {"paths": import_result.get("written_files") or []},
        }
        _emit_kb_event(
            "kb_intake_dialog_completed",
            session=session,
            source="kb-intake.save",
            status="completed",
            transcript=True,
            result=completion_payload,
        )
        try:
            _emit_kb_event(
                "kb_intake_saved",
                session=session,
                source="kb-intake.save",
                status="saved",
                transcript=False,
                result=completion_payload,
            )
        except Exception:
            pass
        return {
            "ok": True,
            "success": True,
            "status": "saved",
            "warnings": plan_warnings,
            "file_path": completion_payload["file_path"],
            "knowledge_item_ids": completion_payload["knowledge_item_ids"],
            "knowledge_node_ids": completion_payload["knowledge_node_ids"],
            "knowledge_edge_ids": completion_payload["knowledge_edge_ids"],
            "hierarchy": completion_payload["hierarchy"],
            "plan_state": plan_state,
            "plan_hash": plan_state.get("plan_hash"),
            "graph_json": graph_doc.model_dump(),
            "vault_write": completion_payload["vault_write"],
            "git": completion_payload["git"],
            "sync": {"mode": "graph_json_import", "new": len(completion_payload["knowledge_item_ids"]), "updated": 0, "error": None},
        }

    persisted_items: list[dict] = []
    persisted_evidence: list[dict] = []
    hierarchy_result: dict[str, Any] = {"items": [], "resolved_links": 0, "missing_links": []}

    try:
        saved_paths = []
        saved_payloads: list[dict] = []
        file_path = None
        for entry in plan_entries:
            path_obj = _write_entry_file(cls["persona_slug"], entry)
            if not path_obj:
                continue
            saved_paths.append(path_obj)
            entry_metadata = {
                **(entry.get("metadata") or {}),
                "slug": entry.get("slug") or _slug_for_plan_entry(entry.get("title") or path_obj.stem),
            }
            saved_payloads.append({
                "title": entry.get("title") or path_obj.stem,
                "content": entry.get("content") or "",
                "content_type": entry.get("content_type") or cls.get("content_type") or "other",
                "tags": entry.get("tags") or [],
                "metadata": entry_metadata,
                "file_path": str(path_obj.relative_to(Path(VAULT_PATH))),
                "file_type": (path_obj.suffix or ".md").lstrip("."),
            })
        file_path = saved_paths[0] if saved_paths else None

        if not file_path:
            _emit_kb_event(
                "kb_intake_dialog_rejected",
                session=session,
                source="kb-intake.save",
                status="rejected",
                transcript=True,
                result={"error": "No files were written."},
            )
            return {"error": "No files were written."}
    except Exception as exc:
        from services import sre_logger
        sre_logger.error("kb_intake", f"Write failed: {exc}", exc)
        _emit_kb_event(
            "kb_intake_dialog_failed",
            session=session,
            source="kb-intake.save",
            status="failed",
            transcript=True,
            result={"error": f"Write failed: {exc}", "failure_type": "write"},
        )
        return {"error": f"Write failed: {exc}"}

    try:
        for payload in saved_payloads:
            item_classification = {
                **{k: v for k, v in cls.items() if k != "file_bytes"},
                "content_type": payload["content_type"],
            }
            persisted = knowledge_lifecycle.persist_pending_knowledge_item(
                persona_slug=cls["persona_slug"],
                title=payload["title"],
                content=payload["content"],
                content_type=payload["content_type"],
                file_path=payload["file_path"],
                file_type=payload["file_type"],
                metadata={
                    **(payload.get("metadata") or {}),
                    "session_id": session_id,
                    "tree_mode": plan_payload.get("tree_mode") or "pyramidal",
                    "branch_policy": plan_payload.get("branch_policy") or "top_down_pyramidal",
                    "sync_origin": "direct_save",
                    "classification": item_classification,
                },
                tags=payload.get("tags") or [],
                source_ref=session_id,
                agent_visibility=["SDR", "Closer", "Classifier"],
            )
            if not persisted or not persisted.get("id"):
                raise RuntimeError(f"Knowledge item was not persisted for {payload['file_path']}")
            persisted_items.append(persisted)
            persisted_evidence.append({
                "knowledge_item_id": persisted.get("id"),
                "knowledge_node_id": (persisted.get("metadata") or {}).get("knowledge_node_id"),
                "status": persisted.get("status"),
                "file_path": persisted.get("file_path"),
                "title": persisted.get("title"),
                "slug": ((persisted.get("metadata") or {}).get("slug")),
            })
        if not persisted_items:
            raise RuntimeError("No knowledge_items were persisted")
        # Hierarchy materialization is best-effort: items are already in the DB,
        # so a transient Supabase glitch here must NOT roll back the whole save.
        # Capture the failure in `hierarchy_result.error` and emit a warning so
        # operators can re-trigger the layout (apply_plan_hierarchy is idempotent).
        try:
            hierarchy_result = knowledge_graph.apply_plan_hierarchy(
                persona_id=persisted_items[0].get("persona_id"),
                persisted_items=persisted_items,
                plan_entries=plan_entries,
                plan_links=plan_payload.get("links") or [],
            )
            try:
                repaired = knowledge_graph.repair_primary_tree_connections(
                    persisted_items[0].get("persona_id"),
                    node_ids=[
                        (item.get("metadata") or {}).get("knowledge_node_id")
                        for item in persisted_items
                        if (item.get("metadata") or {}).get("knowledge_node_id")
                    ],
                )
                hierarchy_result["tree_guard"] = repaired
            except Exception as guard_exc:
                hierarchy_result["tree_guard_error"] = str(guard_exc)
        except Exception as hier_exc:
            from services import sre_logger
            sre_logger.warn(
                "kb_intake",
                f"apply_plan_hierarchy failed (items still persisted): {hier_exc}",
                hier_exc,
            )
            hierarchy_result = {
                "items": [],
                "resolved_links": 0,
                "missing_links": [],
                "error": str(hier_exc),
            }
        hierarchy_by_item = {
            item.get("knowledge_item_id"): item
            for item in hierarchy_result.get("items") or []
            if item.get("knowledge_item_id")
        }
        for evidence in persisted_evidence:
            hierarchy_item = hierarchy_by_item.get(evidence.get("knowledge_item_id")) or {}
            if hierarchy_item.get("main_tree_edge_id"):
                evidence["main_tree_edge_id"] = hierarchy_item.get("main_tree_edge_id")
            if hierarchy_item.get("parent_slug"):
                evidence["parent_slug"] = hierarchy_item.get("parent_slug")
            if hierarchy_item.get("resolution_mode"):
                evidence["resolution_mode"] = hierarchy_item.get("resolution_mode")
            if hierarchy_item.get("quarantine_reason"):
                evidence["quarantine_reason"] = hierarchy_item.get("quarantine_reason")
        for persisted in persisted_items:
            hierarchy_item = hierarchy_by_item.get(persisted.get("id")) or {}
            if not hierarchy_item:
                continue
            updated_metadata = {
                **(persisted.get("metadata") or {}),
                "resolution_mode": hierarchy_item.get("resolution_mode"),
                "quarantine_state": "structural" if hierarchy_item.get("resolution_mode") == "quarantined" else None,
                "quarantine_reason": hierarchy_item.get("quarantine_reason"),
                "resolved_parent_slug": hierarchy_item.get("parent_slug"),
                "resolved_parent_node_id": hierarchy_item.get("parent_node_id"),
            }
            updated_metadata = {k: v for k, v in updated_metadata.items() if v is not None}
            supabase_client.update_knowledge_item(persisted["id"], {"metadata": updated_metadata})
    except Exception as exc:
        from services import sre_logger
        sre_logger.error("kb_intake", f"Persistence failed: {exc}", exc)
        failure_type = "db_persist"
        message = str(exc)
        violations: list[str] = []
        if message.startswith("contract:"):
            failure_type = "db_contract"
            violations = [v.strip() for v in message[len("contract:"):].split(";") if v.strip()]
        elif "insert failed" in message:
            failure_type = "db_insert"
        elif "returned no row" in message:
            failure_type = "db_confirm"
        elif "graph node not confirmed" in message:
            failure_type = "graph_confirm"
        elif "without id" in message:
            failure_type = "db_contract"
        result = {
            "error": f"Persistence failed: {exc}",
            "failure_type": failure_type,
            "saved_paths": [str(p) for p in saved_paths],
        }
        if violations:
            result["violations"] = violations
        _emit_kb_event(
            "kb_intake_dialog_failed",
            session=session,
            source="kb-intake.save",
            status="failed",
            transcript=True,
            result=result,
        )
        response = {"error": f"Persistence failed: {exc}"}
        if violations:
            response["violations"] = violations
        return response

    try:
        git_results = []
        for path_obj in saved_paths:
            rel_p = str(path_obj.relative_to(Path(VAULT_PATH)))
            git_results.append(_git_ops(VAULT_PATH, rel_p, path_obj.name, cls["persona_slug"]))
        git_result = git_results[0] if git_results else {"ok": True, "git": "skipped"}
    except Exception as exc:
        git_result = {
            "add_ok": False,
            "commit_ok": False,
            "push_ok": False,
            "error": f"git unavailable: {exc}".strip()[:200],
        }

    try:
        rel_path = str(file_path.relative_to(Path(VAULT_PATH)))
    except Exception:
        rel_path = file_path.name if file_path else "unknown"

    session["stage"] = "done"
    session["status"] = "saved"
    _save_session(session)
    git_warnings = [
        {
            "stage": "git_push",
            "message": "Knowledge saved, but git push failed.",
            "detail": git_result.get("error"),
        }
    ] if not bool(git_result.get("push_ok", True)) else []
    completion_warnings = [*plan_warnings, *git_warnings]
    completion_payload = {
        "file_path": rel_path,
        "saved_paths": [str(p) for p in saved_paths],
        "git": git_result,
        "success": True,
        "status": "saved_with_warnings" if completion_warnings else "saved",
        "warnings": completion_warnings,
        "sync_mode": "manual_only",
        "entries_written": len(saved_paths),
        "plan_entries": len(plan_entries),
        "plan_links": len(plan_payload.get("links") or []),
        "knowledge_item_ids": [item.get("id") for item in persisted_items],
        "knowledge_node_ids": [
            evidence.get("knowledge_node_id")
            for evidence in persisted_evidence
            if evidence.get("knowledge_node_id")
        ],
        "persistence_evidence": persisted_evidence,
        "hierarchy": hierarchy_result,
        "plan_state": plan_state,
        "plan_hash": plan_state.get("plan_hash"),
        "vault_write": {"paths": [str(p) for p in saved_paths]},
    }
    _emit_kb_event(
        "kb_intake_dialog_completed",
        session=session,
        source="kb-intake.save",
        status="completed",
        transcript=True,
        result=completion_payload,
    )
    # Specific named event for "save successfully landed in DB+graph". Carries
    # the same payload as dialog_completed so subscribers that only watch the
    # specific name don't have to filter by status.
    try:
        _emit_kb_event(
            "kb_intake_saved",
            session=session,
            source="kb-intake.save",
            status="saved",
            transcript=False,
            result=completion_payload,
        )
    except Exception:
        pass

    return {
        "ok": True,
        "success": True,
        "status": "saved_with_warnings" if completion_warnings else "saved",
        "warnings": completion_warnings,
        "file_path": rel_path,
        "knowledge_item_ids": [item.get("id") for item in persisted_items],
        "knowledge_node_ids": [
            evidence.get("knowledge_node_id")
            for evidence in persisted_evidence
            if evidence.get("knowledge_node_id")
        ],
        "persistence_evidence": persisted_evidence,
        "hierarchy": hierarchy_result,
        "plan_state": plan_state,
        "plan_hash": plan_state.get("plan_hash"),
        "vault_write": {"paths": [str(p) for p in saved_paths]},
        "git": git_result,
        "sync": {
            "mode": "manual_only",
            "new": 0,
            "updated": 0,
            "error": None,
        },
    }
