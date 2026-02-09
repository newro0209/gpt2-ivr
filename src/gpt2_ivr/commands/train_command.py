"""학습 커맨드"""

from __future__ import annotations

from typing import Any

from gpt2_ivr.commands.base import Command
from gpt2_ivr.utils.logging_config import get_logger


class TrainCommand(Command):
    """학습 커맨드"""

    def __init__(self) -> None:
        self.logger = get_logger("gpt2_ivr.train")

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """학습 단계를 실행한다."""
        self.logger.info("🚧 train 단계는 아직 구현 중입니다.")
        self.logger.info(
            "향후 `src/gpt2_ivr/training/train.py`를 통해 accelerate 기반 학습을 연결합니다."
        )
        return {
            "status": "not_implemented",
            "message": "train 단계 구현이 필요합니다.",
        }

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "train"
