"""Compatibility alias for the transport-owned database repository.

New code imports ``repositories.transport`` directly. Existing modules keep
receiving the exact same module object while imports are migrated, so test
monkeypatches and the shared client cache retain their original semantics.
"""

from __future__ import annotations

import sys

from repositories import transport as _repository


sys.modules[__name__] = _repository
