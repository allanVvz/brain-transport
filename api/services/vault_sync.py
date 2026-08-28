"""
Vault sync service: scans the local Obsidian vault and creates knowledge_items
with status='pending' for review in the Knowledge Validation queue.
"""
import os
import re
import json
import httpx
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from urllib.parse import quote

from services import supabase_client, knowledge_graph

VAULT_PATH = os.environ.get("VAULT_PATH", r"C:\Ai-Brain\Ai-Brain")
OBSIDIAN_LOCAL_PATH = os.environ.get("OBSIDIAN_LOCAL_PATH") or VAULT_PATH


def _source_mode() -> str:
    return (os.environ.get("VAULT_SOURCE_MODE") or "local").strip().lower()


def _local_vault_path(vault_path: Optional[str] = None) -> str:
    return vault_path or os.environ.get("OBSIDIAN_LOCAL_PATH") or os.environ.get("VAULT_PATH") or VAULT_PATH


def _github_vault_config() -> dict:
    repo = (os.environ.get("GITHUB_VAULT_REPO") or "").strip()
    owner = (os.environ.get("GITHUB_VAULT_OWNER") or "").strip()
    name = (os.environ.get("GITHUB_VAULT_NAME") or "").strip()
    if repo and "/" in repo:
        owner, name = repo.split("/", 1)
    if not owner or not name:
        raise RuntimeError("Configure GITHUB_VAULT_REPO=owner/repo or GITHUB_VAULT_OWNER + GITHUB_VAULT_NAME")
    return {
        "owner": owner,
        "repo": name,
        "branch": (os.environ.get("GITHUB_VAULT_BRANCH") or "main").strip(),
        "root": (os.environ.get("GITHUB_VAULT_ROOT") or "").strip().strip("/"),
        "token": os.environ.get("GITHUB_VAULT_TOKEN") or os.environ.get("GITHUB_TOKEN") or "",
    }


def _github_headers(accept: str = "application/vnd.github+json") -> dict:
    cfg = _github_vault_config()
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-brain-vault-sync",
    }
    if cfg["token"]:
        headers["Authorization"] = f"Bearer {cfg['token']}"
    return headers


def _github_get_json(url: str, headers: dict) -> dict:
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def _github_get_text(url: str, headers: dict) -> str:
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def _vault_source_label(vault_path: Optional[str] = None) -> str:
    if _source_mode() == "github":
        cfg = _github_vault_config()
        root = f"/{cfg['root']}" if cfg["root"] else ""
        return f"github://{cfg['owner']}/{cfg['repo']}@{cfg['branch']}{root}"
    return _local_vault_path(vault_path)


def persona_folder_name(persona_slug: Optional[str]) -> str:
    slug = (persona_slug or "global").strip().lower()
    if slug == "global":
        return "00_GLOBAL"
    return re.sub(r"[^A-Z0-9]+", "_", slug.upper()).strip("_") or "00_GLOBAL"


def persona_vault_dir(persona_slug: Optional[str], vault_path: Optional[str] = None) -> Path:
    root = Path(_local_vault_path(vault_path))
    return root / "AI-BRAIN" / "05_ENTITIES" / "CLIENTS" / persona_folder_name(persona_slug)


def ensure_persona_vault_structure(persona_slug: Optional[str], vault_path: Optional[str] = None) -> dict[str, str]:
    base = persona_vault_dir(persona_slug, vault_path)
    base.mkdir(parents=True, exist_ok=True)
    for folder in (
        "01_BRAND",
        "02_BRIEFING",
        "03_PRODUCTS",
        "04_CAMPAIGNS",
        "05_COPY",
        "06_FAQ",
        "07_TONE",
        "08_AUDIENCE",
        "09_COMPETITORS",
        "10_RULES",
        "11_PROMPTS",
        "12_MAKER",
        "assets",
        "00_OTHER",
    ):
        (base / folder).mkdir(parents=True, exist_ok=True)
    return {
        "persona_slug": persona_slug or "global",
        "folder_name": base.name,
        "path": str(base),
    }


def delete_vault_relative_path(file_path: Optional[str], vault_path: Optional[str] = None) -> bool:
    if not file_path:
        return False
    root = Path(_local_vault_path(vault_path)).resolve()
    target = (root / str(file_path)).resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError("Refusing to delete file outside vault root")
    if not target.exists():
        return False
    target.unlink()
    return True

# Map vault folder names → persona slugs
_FOLDER_TO_SLUG: dict[str, str] = {
    "BAITA_CONVENIENCIA": "baita-conveniencia",
    "TOCK_FATAL": "tock-fatal",
    "PRIME_HIGIENIZACAO": "prime-higienizacao",
    "PRIME": "prime-higienizacao",
    "VZ_LUPAS": "vz-lupas",
    "BAITA": "baita-conveniencia",
    "baita": "baita-conveniencia",
    "tock": "tock-fatal",
    "prime": "prime-higienizacao",
    "vz_lupas": "vz-lupas",
}

_SYNC_ORIGINS_CANONICAL = {"direct_save", "manual_sync"}

# Folders/files to always skip
_SKIP_DIRS = {".obsidian", ".git", "__pycache__", "node_modules", ".trash"}
_SKIP_FILES = {".DS_Store", "Thumbs.db"}

# Text-extractable extensions
_TEXT_EXTS = {".md", ".txt", ".json"}
_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".mp4", ".pdf"}


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown. Returns (frontmatter_dict, body)."""
    if not content.startswith("---"):
        return {}, content
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not match:
        return {}, content
    try:
        import yaml
        fm = yaml.safe_load(match.group(1)) or {}
    except Exception:
        fm = {}
    return fm, content[match.end():]


def _detect_persona(path: Path, frontmatter: dict) -> Optional[str]:
    """Determine persona slug from path or frontmatter."""
    # Explicit in frontmatter
    client_id = frontmatter.get("client_id") or frontmatter.get("cliente") or frontmatter.get("client")
    if client_id:
        normalized = str(client_id).upper().replace("-", "_").replace(" ", "_")
        if normalized in _FOLDER_TO_SLUG:
            return _FOLDER_TO_SLUG[normalized]
        return str(client_id).lower().replace("_", "-")

    # From path parts
    parts_upper = [p.upper().replace(" ", "_") for p in path.parts]
    for folder, slug in _FOLDER_TO_SLUG.items():
        if folder.upper() in parts_upper:
            return slug

    for persona in supabase_client.get_personas() or []:
        slug = persona.get("slug")
        if slug and persona_folder_name(slug) in parts_upper:
            return slug

    # projetos/{client}/ pattern
    for i, part in enumerate(path.parts):
        if part.lower() == "projetos" and i + 1 < len(path.parts):
            sub = path.parts[i + 1].lower()
            if "baita" in sub:
                return "baita-conveniencia"
            if "tock" in sub:
                return "tock-fatal"
            if "vz" in sub or "lupa" in sub:
                return "vz-lupas"

    return None


def _canonical_sync_origin(frontmatter: dict) -> str:
    return str(frontmatter.get("sync_origin") or "").strip().lower()


def _created_via(frontmatter: dict) -> str:
    return str(frontmatter.get("created_via") or "").strip().lower()


def _is_canonical_vault_document(frontmatter: dict, content_type: str) -> bool:
    origin = _canonical_sync_origin(frontmatter)
    created_via = _created_via(frontmatter)
    if origin in _SYNC_ORIGINS_CANONICAL:
        return True
    if created_via == "kb_intake_sofia":
        return True
    slug = str(frontmatter.get("slug") or "").strip()
    parent_slug = str(frontmatter.get("parent_slug") or "").strip()
    if not slug:
        return False
    if content_type in {"faq", "copy", "rule", "tone", "asset"} and not parent_slug:
        return False
    return False


def _quarantine_reason(frontmatter: dict, content_type: str) -> Optional[str]:
    if _is_canonical_vault_document(frontmatter, content_type):
        return None
    slug = str(frontmatter.get("slug") or "").strip()
    if not slug:
        return "missing_slug"
    if content_type in {"faq", "copy", "rule", "tone", "asset"} and not str(frontmatter.get("parent_slug") or "").strip():
        return "missing_parent_slug"
    return "legacy_markdown"


def _detect_content_type(path: Path, frontmatter: dict) -> str:
    """Detect knowledge content type from path + frontmatter."""
    # Explicit frontmatter
    fm_type = frontmatter.get("type") or frontmatter.get("tipo")
    if fm_type:
        type_map = {
            "brand": "brand", "marca": "brand",
            "briefing": "briefing",
            "product": "product", "produto": "product",
            "campaign": "campaign", "campanha": "campaign",
            "copy": "copy",
            "asset": "asset",
            "prompt": "prompt",
            "faq": "faq",
            "maker": "maker_material",
            "tone": "tone", "tom": "tone",
            "competitor": "competitor", "concorrente": "competitor",
            "audience": "audience", "persona": "audience",
            "rule": "rule", "regra": "rule",
        }
        return type_map.get(str(fm_type).lower(), "other")

    name = path.stem.lower()
    parts_lower = [p.lower() for p in path.parts]

    # Asset files
    if path.suffix.lower() in _ASSET_EXTS:
        return "asset"

    # Path-based detection (most reliable)
    if "01_brand_positioning" in name or name == "brand":
        return "brand"
    if "briefing" in name:
        return "briefing"
    if "tone" in name or "tom" in name or "07_tone" in parts_lower:
        return "tone"
    if "audience" in name or "persona_buyer" in name or "08_audience" in parts_lower:
        return "audience"
    if "moodboard" in name or "12_maker" in parts_lower:
        return "maker_material"
    if "competitor" in parts_lower or "competitors" in parts_lower or "09_competitors" in parts_lower or name in {"nina_luxo", "supervaidosa", "zafira"}:
        return "competitor"
    if "07_prompts" in parts_lower or "prompts" in parts_lower:
        return "prompt"
    if "01_rules" in parts_lower or "rules" in parts_lower:
        return "rule"
    if "06_patterns" in parts_lower or "patterns" in parts_lower:
        return "rule"
    if "06_intents" in parts_lower or "intents" in parts_lower:
        return "rule"
    if "campanha" in parts_lower or "campanhas" in parts_lower or "04_campaigns" in parts_lower or "campaign" in name:
        return "campaign"
    if "copie" in name or "copy" in name or "copies" in parts_lower or "05_copy" in parts_lower:
        return "copy"
    if "produto" in name or "products" in name or "03_products" in parts_lower:
        return "product"
    if "faq" in name or "kb" in name or "06_faq" in parts_lower:
        return "faq"
    if "01_brand" in parts_lower:
        return "brand"
    if "02_briefing" in parts_lower:
        return "briefing"
    if "10_rules" in parts_lower:
        return "rule"
    if "11_prompts" in parts_lower:
        return "prompt"
    if "02_skills" in parts_lower:
        return "prompt"
    if "04_memory" in parts_lower:
        return "other"
    if "projetos" in parts_lower:
        return "campaign"

    return "other"


def _file_title(path: Path, frontmatter: dict) -> str:
    """Generate a human-readable title for the knowledge item."""
    if frontmatter.get("title"):
        return str(frontmatter["title"])
    # Use stem, convert snake_case to title
    stem = path.stem.replace("_", " ").replace("-", " ")
    return stem.title()


def _should_skip(path: Path) -> bool:
    """Return True if file/dir should be excluded from sync."""
    for part in path.parts:
        if part in _SKIP_DIRS:
            return True
    if path.name in _SKIP_FILES:
        return True
    # Skip Obsidian internal files
    if path.suffix == ".md" and path.name.startswith("."):
        return True
    return False


def _iter_local_vault_files(vault_path: Optional[str] = None):
    root = Path(_local_vault_path(vault_path))
    if not root.exists():
        raise FileNotFoundError(f"Vault path not found: {root}")
    for fp in root.rglob("*"):
        if fp.is_dir() or _should_skip(fp):
            continue
        ext = fp.suffix.lower()
        if ext not in _TEXT_EXTS and ext not in _ASSET_EXTS:
            continue
        if ext in _TEXT_EXTS:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        else:
            raw = f"[asset: {fp.name}]"
        yield fp, raw, root


def _iter_github_vault_files():
    cfg = _github_vault_config()
    owner = quote(cfg["owner"], safe="")
    repo = quote(cfg["repo"], safe="")
    branch = quote(cfg["branch"], safe="")
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    tree = _github_get_json(tree_url, _github_headers())
    root_prefix = f"{cfg['root']}/" if cfg["root"] else ""
    api_root = f"https://api.github.com/repos/{owner}/{repo}/contents"

    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        full_path = item.get("path") or ""
        if root_prefix and not full_path.startswith(root_prefix):
            continue
        rel_path = full_path[len(root_prefix):] if root_prefix else full_path
        fp = Path(rel_path)
        if _should_skip(fp):
            continue
        ext = fp.suffix.lower()
        if ext not in _TEXT_EXTS and ext not in _ASSET_EXTS:
            continue
        if ext in _TEXT_EXTS:
            contents_url = f"{api_root}/{quote(full_path, safe='/')}?ref={branch}"
            raw = _github_get_text(contents_url, _github_headers("application/vnd.github.raw"))
        else:
            raw = f"[asset: {fp.name}]"
        yield fp, raw, Path(".")


def _yield_vault_files(vault_path: Optional[str] = None):
    """Yield (path, raw_content, root) from the configured vault source.

    GitHub mode uses the GitHub API and does not clone or read a local
    repository. Local mode preserves the previous filesystem scan.
    """
    if _source_mode() == "github":
        yield from _iter_github_vault_files()
        return
    yield from _iter_local_vault_files(vault_path)


def _frontmatter_and_body(path: Path, raw: str) -> tuple[dict, str]:
    ext = path.suffix.lower()
    if ext == ".md":
        return _parse_frontmatter(raw)
    if ext == ".json":
        try:
            parsed = json.loads(raw)
            fm = {"cliente": parsed.get("cliente", "")} if isinstance(parsed, dict) else {}
        except Exception:
            fm = {}
        return fm, raw
    return {}, raw


def scan_vault(vault_path: str = VAULT_PATH) -> dict:
    """
    Scan the vault and return a preview of what would be synced.
    Returns: {files: [{path, persona, content_type, title}], total, by_client, by_type}
    """
    files = []
    try:
        vault_files = list(_yield_vault_files(vault_path))
    except Exception as exc:
        return {"error": str(exc), "files": [], "source_mode": _source_mode()}

    for fp, raw, root in vault_files:
        ext = fp.suffix.lower()
        fm = {}
        if ext in _TEXT_EXTS:
            try:
                fm, _ = _frontmatter_and_body(fp, raw)
            except Exception:
                pass

        persona_slug = _detect_persona(fp, fm)
        content_type = _detect_content_type(fp, fm)

        files.append({
            "path": fp.relative_to(root).as_posix(),
            "full_path": str(fp),
            "persona": persona_slug,
            "content_type": content_type,
            "title": _file_title(fp, fm),
            "ext": ext,
        })

    by_client: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for f in files:
        k = f["persona"] or "unassigned"
        by_client[k] = by_client.get(k, 0) + 1
        by_type[f["content_type"]] = by_type.get(f["content_type"], 0) + 1

    return {
        "files": files,
        "total": len(files),
        "by_client": by_client,
        "by_type": by_type,
        "source_mode": _source_mode(),
        "source": _vault_source_label(vault_path),
    }


def run_sync(vault_path: str = VAULT_PATH, persona_filter: Optional[str] = None) -> dict:
    """
    Execute a full vault sync: create/update knowledge_items with status=pending.
    Returns summary with run_id and counts.
    """
    try:
        vault_files = list(_yield_vault_files(vault_path))
    except Exception as exc:
        return {"error": str(exc)}

    # Get or create the vault knowledge source
    source_path = _vault_source_label(vault_path)
    source = supabase_client.get_knowledge_source_by_path(source_path)
    if not source:
        source = supabase_client.insert_knowledge_source({
            "source_type": "vault",
            "name": "Brain AI Vault",
            "path": source_path,
        })

    source_id = source["id"]

    # Create sync run
    run = supabase_client.insert_sync_run({"source_id": source_id, "status": "running"})
    run_id = run["id"]

    counts = {
        "found": 0,
        "new": 0,
        "updated": 0,
        "skipped": 0,
        "imported_canonical": 0,
        "imported_quarantined": 0,
        "skipped_legacy": 0,
    }
    persona_cache: dict[str, Optional[str]] = {}  # slug → id

    def _get_persona_id(slug: Optional[str]) -> Optional[str]:
        if not slug:
            return None
        if slug not in persona_cache:
            p = supabase_client.get_persona(slug)
            persona_cache[slug] = p["id"] if p else None
        return persona_cache[slug]

    for fp, raw, root in vault_files:
        ext = fp.suffix.lower()

        counts["found"] += 1
        rel_path = fp.relative_to(root).as_posix()

        try:
            fm, body = _frontmatter_and_body(fp, raw)

            persona_slug = _detect_persona(fp, fm)
            content_type = _detect_content_type(fp, fm)
            title = _file_title(fp, fm)

            # Apply persona filter if specified
            if persona_filter and persona_slug != persona_filter:
                counts["skipped"] += 1
                supabase_client.insert_sync_log({
                    "run_id": run_id, "file_path": rel_path,
                    "action": "skipped", "content_type": content_type,
                })
                continue

            persona_id = _get_persona_id(persona_slug)
            quarantine_reason = _quarantine_reason(fm, content_type)
            sync_origin = "manual_sync"
            quarantine_state = "legacy_orphaned" if quarantine_reason else None

            # Check if already exists
            existing = supabase_client.get_knowledge_item_by_path(rel_path)

            # Keep writes compatible with older Supabase constraints that only
            # accept pending/approved/rejected/embedded. Missing persona/category
            # is still visible through metadata and the queue's pending status.
            auto_status = "pending"

            item_data = {
                "persona_id": persona_id,
                "source_id": source_id,
                "status": auto_status,
                "content_type": content_type,
                "title": title,
                "content": body[:8000],
                "metadata": {
                    **{k: v for k, v in fm.items() if isinstance(v, (str, int, float, bool, list))},
                    "needs_persona": persona_id is None,
                    "needs_category": content_type == "other",
                    "source_mode": _source_mode(),
                    "vault_source": source_path,
                    "sync_origin": sync_origin,
                    "quarantine_state": quarantine_state,
                    "quarantine_reason": quarantine_reason,
                },
                "file_path": rel_path,
                "file_type": ext.lstrip("."),
            }

            if existing:
                # Only update if content changed
                if existing["content"] != item_data["content"]:
                    item_data["status"] = existing.get("status", "pending")  # preserve approval state
                    supabase_client.update_knowledge_item(existing["id"], item_data)
                    counts["updated"] += 1
                    action = "updated_quarantined" if quarantine_reason else "updated"
                    item_for_graph = {**existing, **item_data}
                else:
                    counts["skipped"] += 1
                    action = "skipped_legacy" if quarantine_reason else "skipped"
                    item_for_graph = existing
            else:
                inserted = supabase_client.insert_knowledge_item(item_data)
                counts["new"] += 1
                action = "created_quarantined" if quarantine_reason else "created"
                item_for_graph = {**item_data, **(inserted or {})}

            if quarantine_reason:
                if action.startswith("skipped"):
                    counts["skipped_legacy"] += 1
                else:
                    counts["imported_quarantined"] += 1
            elif action != "skipped":
                counts["imported_canonical"] += 1

            supabase_client.insert_sync_log({
                "run_id": run_id,
                "file_path": rel_path,
                "persona_id": persona_id,
                "action": action,
                "content_type": content_type,
                "error_message": quarantine_reason,
            })

            # Mirror into the semantic graph (knowledge_nodes/edges).
            # Defensive: never let graph maintenance break the sync.
            try:
                if item_for_graph and item_for_graph.get("id") and not quarantine_reason:
                    knowledge_graph.bootstrap_from_item(
                        item_for_graph,
                        frontmatter=fm,
                        body=body,
                        persona_id=persona_id,
                        source_table="knowledge_items",
                    )
            except Exception as exc:
                supabase_client.insert_sync_log({
                    "run_id": run_id,
                    "file_path": rel_path,
                    "action": "error",
                    "error_message": f"Graph mirror warning: {str(exc)[:470]}",
                })

        except Exception as e:
            counts["skipped"] += 1
            supabase_client.insert_sync_log({
                "run_id": run_id,
                "file_path": rel_path,
                "action": "error",
                "error_message": str(e)[:500],
            })

    # Finalize run
    from datetime import datetime, timezone
    from services.event_emitter import emit as _emit
    supabase_client.update_sync_run(run_id, {
        "status": "completed",
        "files_found": counts["found"],
        "files_new": counts["new"],
        "files_updated": counts["updated"],
        "files_skipped": counts["skipped"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    supabase_client.update_knowledge_source(source_id, {
        "last_synced_at": datetime.now(timezone.utc).isoformat()
    })
    _emit("vault_sync_completed", entity_type="sync_run", entity_id=run_id, payload=counts)
    return {"run_id": run_id, **counts}
