"""Create the optional first production admin, without a known default login."""
from __future__ import annotations

import os

from create_auth_user import main as create_auth_user


if __name__ == "__main__":
    if not (os.environ.get("AI_BRAIN_SEED_ADMIN_EMAIL") or "").strip():
        print("AI_BRAIN_SEED_ADMIN_EMAIL is empty; initial admin seed skipped.")
    elif not (os.environ.get("AI_BRAIN_SEED_ADMIN_PASSWORD") or "").strip():
        raise SystemExit("AI_BRAIN_SEED_ADMIN_PASSWORD is required when seeding an admin.")
    else:
        create_auth_user()

