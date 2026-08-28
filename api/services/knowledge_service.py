import os
import json
import gspread
from google.oauth2.service_account import Credentials
from services import supabase_client
from typing import Optional

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def _get_sheet(service_account_json: Optional[str] = None):
    if service_account_json:
        creds = Credentials.from_service_account_info(json.loads(service_account_json), scopes=_SCOPES)
    else:
        creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "secrets/google-service-account.json")
        creds = Credentials.from_service_account_file(creds_path, scopes=_SCOPES)
    return gspread.authorize(creds)


def sync_from_sheets(
    persona_id: str,
    spreadsheet_id: Optional[str] = None,
    *,
    service_account_json: Optional[str] = None,
) -> int:
    sid = spreadsheet_id or os.environ["KB_SPREADSHEET_ID"]
    gc = _get_sheet(service_account_json=service_account_json)
    sheet = gc.open_by_key(sid).worksheet("FAQ")
    rows = sheet.get_all_records()

    count = 0
    for i, row in enumerate(rows):
        if not row.get("titulo") or not row.get("conteudo"):
            continue

        kb_id = str(row.get("kb_id") or f"row_{i + 2}")
        entry = {
            "kb_id": kb_id,
            "persona_id": persona_id,
            "tipo": (row.get("tipo") or "faq").lower(),
            "categoria": row.get("categoria") or "geral",
            "produto": row.get("produto") or "geral",
            "intencao": row.get("intencao") or "duvida_geral",
            "titulo": row["titulo"],
            "conteudo": row["conteudo"],
            "link": row.get("link") or None,
            "prioridade": int(row.get("prioridade") or 99),
            "status": row.get("status") or "ATIVO",
            "source": "sheets",
        }
        supabase_client.upsert_kb_entry(entry)
        count += 1

    return count


def search_kb_text(query: str, persona_id: Optional[str] = None, top_k: int = 5) -> list[str]:
    if not persona_id:
        # Never allow a global fallback that could mix clients.
        return []
    try:
        rag_chunks = supabase_client.search_active_rag_chunks(
            persona_id=persona_id,
            query=query,
            limit=top_k,
        )
    except Exception:
        rag_chunks = []
    if rag_chunks:
        return [
            str(chunk.get("chunk_text") or chunk.get("chunk_summary") or "").strip()
            for chunk in rag_chunks
            if str(chunk.get("chunk_text") or chunk.get("chunk_summary") or "").strip()
        ][:top_k]

    # Temporary, auditable transition fallback.
    try:
        supabase_client.insert_event(
            {
                "event_type": "legacy_kb_fallback_used",
                "entity_type": "persona",
                "entity_id": persona_id,
                "persona_id": persona_id,
                "payload": {"query": (query or "")[:300], "top_k": top_k},
                "level": "warning",
                "source": "knowledge_service.search_kb_text",
            },
            source="knowledge_service.search_kb_text",
        )
    except Exception:
        pass
    entries = supabase_client.get_kb_entries(persona_id=persona_id)
    query_lower = query.lower()
    scored = []
    for e in entries:
        text = f"{e.get('titulo','')} {e.get('conteudo','')} {e.get('categoria','')} {e.get('produto','')}".lower()
        score = sum(1 for word in query_lower.split() if word in text)
        if score > 0:
            content = (
                f"Pergunta: {e['titulo']}\n"
                f"Resposta: {e['conteudo']}"
                + (f"\nLink: {e['link']}" if e.get("link") else "")
            )
            scored.append((score, content))

    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]

