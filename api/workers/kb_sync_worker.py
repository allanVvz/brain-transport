import os
from services import supabase_client, sre_logger
from services.knowledge_service import sync_from_sheets
from workers.base_worker import BaseWorker

_CREDS_PATH = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "secrets/google-service-account.json")


class KbSyncWorker(BaseWorker):
    name = "KbSyncWorker"
    interval = int(os.environ.get("KB_SYNC_INTERVAL", 3600))

    def _run_cycle(self):
        if not os.path.exists(_CREDS_PATH):
            sre_logger.warn(
                self.name,
                f"Google credentials not found at '{_CREDS_PATH}' â€” standby, skipping sync",
            )
            return

        personas = supabase_client.get_personas()
        for persona in personas:
            spreadsheet_id = (persona.get("config") or {}).get("kb_spreadsheet_id")
            if not spreadsheet_id:
                continue
            try:
                count = sync_from_sheets(
                    persona_id=persona["id"],
                    spreadsheet_id=spreadsheet_id,
                )
                sre_logger.info(self.name, f"persona={persona['slug']} synced={count} entries")
            except FileNotFoundError as exc:
                sre_logger.warn(self.name, f"credentials missing mid-run for persona={persona['slug']}", exc)
            except Exception as exc:
                sre_logger.error(self.name, f"sync failed for persona={persona['slug']}", exc)

