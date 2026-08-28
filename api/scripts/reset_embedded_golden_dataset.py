import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from services import supabase_client


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Deactivate legacy Embedded links and clear Golden Dataset / RAG mirrors."
    )
    parser.add_argument("--persona-id", default=None, help="Optional persona_id to scope the reset.")
    args = parser.parse_args()

    report = supabase_client.reset_embedded_legacy_publications(persona_id=args.persona_id)
    print(report)


if __name__ == "__main__":
    main()
