"""Tokenizer Remapping Command"""
from __future__ import annotations

from typing import Any

from gpt2_ivr.commands.base import Command


class RemapCommand(Command):
    """Remap Command"""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        print("Executing Remap Command...")
        # TODO: Implement remap logic using `src/gpt2_ivr/tokenizer/remap_rules.yaml`
        # and create the remapped tokenizer in `artifacts/tokenizers/remapped/`
        print("Remap Command finished.")
        return {}

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "remap"
