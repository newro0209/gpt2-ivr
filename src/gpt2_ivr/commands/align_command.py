"""Embedding Alignment Command"""

from __future__ import annotations

from typing import Any

from gpt2_ivr.commands.base import Command


class AlignCommand(Command):
    """Align Command"""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        print("Executing Align Command...")
        # TODO: Implement embedding alignment logic.
        # This should use logic from `src/gpt2_ivr/embedding/`
        print("Align Command finished.")
        return {}

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "align"
