from __future__ import annotations

import argparse
import asyncio
import signal

from workers.health_check_worker import HealthCheckWorker
from workers.media_ingest_worker import MediaIngestWorker
from workers.whatsapp_dispatch_worker import WhatsAppDispatchWorker


WORKERS = {
    "health_check": HealthCheckWorker,
    "whatsapp_dispatch": WhatsAppDispatchWorker,
    "media_ingest": MediaIngestWorker,
}


async def _run(selected: list[str]) -> None:
    workers = [WORKERS[name]() for name in selected]
    tasks = [asyncio.create_task(worker.start()) for worker in workers]
    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()

    def request_shutdown() -> None:
        if stopping.is_set():
            return
        stopping.set()
        for worker in workers:
            worker.request_stop()

    for signame in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signame, request_shutdown)
        except NotImplementedError:  # Windows test runner
            pass
    try:
        await asyncio.gather(*tasks)
    finally:
        request_shutdown()
        await asyncio.gather(
            *(worker.wait_for_drain() for worker in workers),
            return_exceptions=True,
        )
        for task in tasks:
            if not task.done():
                task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Brain AI background workers.")
    parser.add_argument("--all", action="store_true", help="Run all workers.")
    parser.add_argument(
        "--worker",
        action="append",
        choices=sorted(WORKERS.keys()),
        help="Run one or more specific workers.",
    )
    args = parser.parse_args()

    selected = sorted(WORKERS.keys()) if args.all or not args.worker else args.worker
    asyncio.run(_run(selected))


if __name__ == "__main__":
    main()
