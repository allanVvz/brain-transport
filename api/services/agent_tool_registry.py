from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from schemas.agent_harness import ToolEffect, ToolRisk


ToolHandler = Callable[..., dict[str, Any] | BaseModel]


@dataclass(frozen=True)
class ToolManifest:
    name: str
    version: str
    owner_agent: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    effect: ToolEffect
    risk: ToolRisk
    permission: str
    persona_scoped: bool
    timeout_seconds: int
    idempotent: bool
    sensitive_fields: tuple[str, ...] = field(default_factory=tuple)
    handler: ToolHandler | None = None
    deprecated: bool = False

    @property
    def schema_checksum(self) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "input": self.input_model.model_json_schema(),
            "output": self.output_model.model_json_schema(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def model_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    def capability(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "owner_agent": self.owner_agent,
            "description": self.description,
            "effect": self.effect.value,
            "risk": self.risk.value,
            "permission": self.permission,
            "persona_scoped": self.persona_scoped,
            "timeout_seconds": self.timeout_seconds,
            "idempotent": self.idempotent,
            "schema_checksum": self.schema_checksum,
        }

    def invoke(self, context: dict[str, Any], arguments: dict[str, Any] | None) -> dict[str, Any]:
        if self.handler is None:
            raise RuntimeError(f"tool {self.name} has no handler")
        parsed = self.input_model.model_validate(arguments or {})
        result = self.handler(context, **parsed.model_dump(exclude_none=True))
        validated = self.output_model.model_validate(
            result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        )
        return validated.model_dump(mode="json")


class ToolRegistry:
    def __init__(self, manifests: list[ToolManifest] | None = None) -> None:
        self._items: dict[str, ToolManifest] = {}
        for manifest in manifests or []:
            self.register(manifest)

    def register(self, manifest: ToolManifest) -> None:
        if manifest.name in self._items:
            raise ValueError(f"duplicate tool name: {manifest.name}")
        self._items[manifest.name] = manifest

    def get(self, name: str) -> ToolManifest | None:
        return self._items.get(name)

    def all(self, *, model_visible_only: bool = False) -> list[ToolManifest]:
        values = list(self._items.values())
        if model_visible_only:
            values = [item for item in values if not item.deprecated]
        return values

    def validate(self) -> None:
        seen: set[str] = set()
        for manifest in self._items.values():
            if not manifest.name.strip() or manifest.name in seen:
                raise ValueError(f"invalid or duplicate tool name: {manifest.name!r}")
            seen.add(manifest.name)
            if not manifest.version.strip() or manifest.handler is None:
                raise ValueError(f"tool {manifest.name} requires version and handler")
            if manifest.permission not in {"view", "edit", "manage"}:
                raise ValueError(f"tool {manifest.name} has invalid permission")
            if manifest.timeout_seconds <= 0:
                raise ValueError(f"tool {manifest.name} requires a positive timeout")
            if manifest.effect in {ToolEffect.WRITE, ToolEffect.DESTRUCTIVE, ToolEffect.EXTERNAL} and not manifest.idempotent:
                raise ValueError(f"mutating tool {manifest.name} must declare idempotency")
            manifest.input_model.model_json_schema()
            manifest.output_model.model_json_schema()

    def model_schemas(self) -> list[dict[str, Any]]:
        self.validate()
        return [item.model_schema() for item in self.all(model_visible_only=True)]

    def capabilities(self) -> list[dict[str, Any]]:
        self.validate()
        return [item.capability() for item in self.all(model_visible_only=True)]

