"""Training Command"""
from __future__ import annotations

from typing import Any

from gpt2_ivr.commands.base import Command


class TrainCommand(Command):
    """Train Command"""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        print("Executing Train Command...")
        # TODO: Implement training logic using `accelerate`
        # This should use `src/gpt2_ivr/training/train.py`
        print("Train Command finished.")
        return {}

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "train"
