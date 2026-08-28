import asyncio
from typing import Callable, Any

_queue: asyncio.Queue = asyncio.Queue()


async def enqueue(item: Any) -> None:
    await _queue.put(item)


async def start_worker(handler: Callable) -> None:
    while True:
        item = await _queue.get()
        try:
            await handler(item)
        except Exception as e:
            print(f"[queue] error: {e}")
        finally:
            _queue.task_done()
