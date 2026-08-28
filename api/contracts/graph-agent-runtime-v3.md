# Graph Agent Runtime v3 — contrato executável

Contract-ID: `graph-agent-runtime-v3`

Compiler: `graph-compiler-v3.6.2`

Este Markdown faz parte da proveniência de cada publicação. O compilador grava
seu caminho e checksum no Graph JSON; qualquer alteração deliberada neste
contrato produz um novo checksum de publicação.

## Autoridade

`knowledge_nodes` e `knowledge_edges` são a única fonte editável. O Graph JSON
v3 é um snapshot compilado, imutável e auditável. Um node só é branch anchor
quando declara `capabilities.branch_anchor=true`.

## Invariantes de compilação

- Um caminho primário não pode ter ciclo nem dois pais.
- Cada field declara `validation.mode`: `enum`, `schema` ou `semantic`.
- `enum` exige valores canônicos e aliases publicados; `schema` exige JSON
  Schema; `semantic` exige descrição e tipo, exemplos ou regras suficientes.
- `scope=declaration` faz o field pertencer ao node que o declarou. Os scopes
  compatíveis `persona` e `branch` permanecem disponíveis.
- Fields obrigatórios precisam de pergunta publicada, statuses aceitos e
  overwrite policy. Dependências de fields não podem formar ciclos.
- Handoff e claims comerciais exigem regra, política e evidência publicada.
- Uma publicação só ativa depois de coordenadas, memberships, contratos,
  entries, chunks e embeddings completos nas quantidades do manifesto.
- O provider de embeddings vem do runtime; o modelo efetivo integra o checksum.
- Claims, rules, validators, perguntas e fatos estruturados possuem chunks
  próprios. Perguntas ancoradas em outro galho não entram no contrato.
- `Embedded` e `Gallery` protegidos são reutilizados pela projeção; versões
  anteriores são desativadas sem exclusão destrutiva.

## Turno conversacional

Antes do turno, `embed` e `embedded` compilam para o mesmo tipo protegido e a
subarvore de um node `global_context` integra todas as memberships de branch.
Publicacoes com `faq_projection_contract=v1` so ativam quando cada FAQ factual
elegivel possui membership, entry e chunk canonico `faq` com pergunta e resposta.

```text
inbound canônico
→ resolução literal/semântica de todos os serviços
→ consumo dos spans de serviço
→ retrieval híbrido dentro da membership
→ proposta JSON estrita do modelo
→ validação declarativa dos demais fields
→ proof checker e um repair direcionado opcional
→ pergunta/fallback publicado
→ ledger + proof exatamente uma vez
→ outbox idempotente
```

`service_operations[]` é o contrato autoritativo do conjunto de serviços. Cada
operação contém `add`, `keep` ou `drop`, anchor publicado, checksum do caminho e
evidência literal. Um novo serviço é adicionado por padrão; somente linguagem
explícita de troca ou remoção gera `drop`. Repetição só muda o foco. Seleção
ambígua não altera ledger nem branches.

Um span consumido como serviço nunca pode validar outro field. O
`active_branch_node_id` representa o foco e `active_branch_node_ids` representa
o conjunto autoritativo. Os campos singulares continuam apenas como adaptador
de compatibilidade.

Na primeira resposta incompatível, nenhum fato é persistido e a pergunta
publicada é repetida. Na segunda, o field recebe `status=unknown`, `value=null`
e motivo `ignored_twice`. “Não sei” explícito pode gerar `unknown`
imediatamente. `collection_complete` encerra a coleta; `qualification_complete`
só é verdadeiro quando todos os campos obrigatórios são conhecidos.

Falha técnica não produz copy nem handoff. Handoff só é válido por regra do
contrato compilado. Claims usam apenas evidência autorizada. Mudanças de
publicação invalidam fatos incompatíveis antes do próximo commit.

## Projeção e auditoria

A nota comercial estruturada separa fatos comuns de fatos por serviço e é
persistida em metadata. `commercial_note` e `interesse_produto` singulares
continuam refletindo o serviço em foco por compatibilidade.

Em mensagens compostas, a clausula interrogativa e ranqueada separadamente.
Pergunta ou alias normalizado exato vence; selecao semantica exige score minimo
de `0.18` e margem de `0.03`. A resposta canonica do FAQ precede exatamente a
primeira pergunta pendente e seu chunk reservado nao participa do MMR.

`canonical_inbound_id` é único em `conversation_turn_proofs`. Fatos preservam
mensagem, evidence span, confiança, revisão e supersessão. O proof expõe a
resolução de serviços, operações, spans consumidos, conjunto anterior/posterior
e resultado de validação de cada field.

Uma resposta HTTP perdida depois do commit não autoriza replay. A reconciliação
operacional só encerra o inbound quando encontra exatamente uma prova válida,
um outbound único e sua mensagem persistida; ela não chama retrieval, modelo
ou transporte.

## Máquina de estados SDR

- `collecting` resolve um serviço publicado e coleta somente o primeiro field
  realmente pendente.
- Campos completos produzem o resumo e a `confirmation_question` publicada,
  transitando para `awaiting_confirmation` sem handoff.
- Somente uma confirmação explícita em turno posterior produz
  `qualified_confirmed`, `route=HUMAN` e handoff no mesmo commit.
- Negação ou correção volta à coleta sem apagar fatos; uma dúvida é respondida
  com FAQ aprovada antes de retomar a confirmação.
- Handoff incompleto registra os fields não confirmados e preserva todos os
  fatos para o humano.
- Depois de reativação, `post_qualification_support` responde saudação e FAQ
  sem reiniciar o roteiro. Uma nova confirmação só é exigida após alteração
  explícita do pedido.

Saudação é intenção transversal do turno atual e nunca preenche field. O field
`servico` é referencial: seu valor, owner, anchor e path checksum precisam
resolver para uma branch da publicação por título, slug, alias ou evidência
semântica segura. Saudações, confirmações, respostas sociais e números isolados
nunca são serviço.

Nenhum turno de coleta é entregue como reconhecimento isolado. Se a supressão
de repetição retirar a pergunta e ainda houver field perguntável, a pergunta
publicada volta e `repetition_action` fica
`repaired_never_acknowledge_only` — a emissão conta no orçamento porque o
cliente a recebeu. Copy de confirmação pode ser publicada como lista de
formulações equivalentes; o runtime escolhe uma que esta conversa ainda não
ouviu.

Uma operação de branch recusada invalida a mutação, não o turno: o conteúdo
declarativo do modelo e a resposta publicada sobrevivem, apenas a alegação de
mudança é descartada.

Ao religar a IA, o agente só fala enquanto a última mensagem não respondida do
cliente for mais recente que
`conversation_policy.reactivation.answer_pending_inbound_within_seconds`.
Passada a janela, os inbounds estacionados permanecem estacionados e o próximo
turno é do cliente. Disparo de campanha e conversa nova são portas distintas e
não passam por essa janela.

A proteção antirrepetição não lança exceção em produção: fatos aceitos são
commitados e o outbound duplicado é suprimido. A supressão retém a pergunta
repetida, não o turno: quando o turno tem conteúdo próprio — dúvida respondida,
serviço reconhecido — esse conteúdo é entregue sem a pergunta e
`repetition_action` é `suppressed_duplicate_question`. Enquanto a IA estiver
ligada, o turno nunca termina em silêncio total por repetição: um handoff
terminal repetido continua suprimido (`suppressed_duplicate_terminal`, nada de
novo a dizer por definição), mas uma pergunta não-terminal repetida sem
conteúdo declarativo próprio ainda assim é entregue — `repetition_action` fica
`allowed_never_silent` e a resposta original (a pergunta) é mantida em vez de
zerada. Uma pergunta suprimida não entra em `asked_question_node_ids`, porque o
orçamento conta emissões entregues. Uma dúvida respondida pelo grafo não é uma
não-resposta e não consome tentativa; uma dúvida apenas adiada consome. CI e
Validator continuam reprovando o critério semântico pelo proof. Todo proof
expõe `intent_audit`, `service_resolution`, `journey_transition`,
`confirmation_state` e `repetition_action`.

## Nome completo e confirmação de serviço

O Graph JSON publica `common_contract` para os fields compartilhados antes da
seleção do primeiro serviço. Ele também publica em `claims` as políticas que
**todos** os branch contracts autorizam — na prática as FAQs projetadas de
`global_context` — junto com os nodes de evidência correspondentes no closure.
Uma claim autorizada por todo galho não depende de qual serviço será escolhido,
então vale durante a descoberta: o cliente que compra perguntando recebe a
resposta publicada antes de escolher. Preço, agenda e regra de um serviço
específico ficam de fora, porque um único galho os declara. O contrato de
pré-seleção do runtime herda exatamente essa lista.

Um field com `validation.semantic_type=human_full_name` aceita de dois a seis
tokens Unicode por padrão — partículas, hífen e apóstrofo incluídos — e o grafo
ajusta esses limites por field com `validation.min_tokens` e
`validation.max_tokens`.

A inferência do modelo é a primeira fonte do valor. O field vira `known`
quando a confiança publicada alcança `validation.model_confidence_min`
(default `0.90`), a forma é um nome válido, o `evidence_span` é reencontrável
na mensagem depois de normalizar caixa, acentos e pontuação, e esse span não
sobrepõe span reservado de serviço. O span persistido é o recorte literal da
mensagem do cliente, nunca a reescrita do modelo. Mensagem composta não impede
nada: o nome é gravado e os demais spans seguem disponíveis para os outros
fields.

Abaixo do piso de confiança, o fallback determinístico aceita a resposta
integral à pergunta publicada de nome, comparada de forma normalizada — caixa,
acento e pontuação não decidem. Só quando modelo e fallback falham é que o
field vira `needs_confirmation`; candidato e proveniência ficam em
`conversation_facts.metadata.confirmation`. Com
`validation.confirmation_policy=last_resort`, o turno entrega primeiro o
conteúdo útil — dúvida respondida, serviço reconhecido, conteúdo declarativo
do modelo — e só então anexa a confirmação publicada.

Um candidato pendente é resolvido por confirmação explícita **ou** pela
repetição do próprio valor: mensagem inteira equivalente ao candidato
normalizado confirma. Rejeição explícita continua invalidando.

`validation` aceita, opcionalmente, `model_confidence_min` (0..1),
`min_tokens`, `max_tokens` e `confirmation_policy`
(`always` | `last_resort`). Declaração inválida reprova a publicação com
`field_validation_tuning_invalid`.

`service_observations[]` é não autoritativo. O backend é o único produtor das
operações aplicadas em `service_operations[]`, e cada operação exige evidência
consumida `exact_catalog` ou `confirmed_candidate`. Exato significa igualdade
depois de normalizar caixa, acentos e pontuação de título, slug ou alias
publicado. Menção exata em dúvida informativa não altera branches.

Resolução textual usa distância Levenshtein máxima `3`, similaridade mínima
`0.80` e candidato único. Resolução semântica consulta somente chunks dos
anchors de serviço e exige cosseno `>=0.78`, margem `>=0.08`, coincidência do
modelo com o ranking do backend e span literal não reservado. Aproximações
produzem apenas `needs_confirmation`; empate usa o template publicado de
desambiguação. Confirmações de nome ou serviço são consumidas antes da
confirmação final da jornada.

## Desfecho comercial

O SDR termina na qualificação. Conversão, venda, entrega e cancelamento são
eventos humanos registrados em `POST /agents/leads/{lead_ref}/journey-events`,
nunca inferidos pelo modelo, e levam a jornada a `converted` ou `closed`.

A projeção a partir do proof **não regride** uma jornada em `converted` ou
`closed`: `journey_transition` continua sendo emitido e a metadata continua
evoluindo, mas o `state` fica com o desfecho registrado pelo humano. Um inbound
depois da venda é suporte ao pedido, não uma nova coleta.

O contrato completo — eventos, idempotência, derivação de `journey_outcome` e a
paleta `resultado/*` — está em `docs/architecture/SDR_JOURNEY_STATE_MACHINE.md`.
