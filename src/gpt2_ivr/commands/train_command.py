"""학습 커맨드.

재할당 토크나이저와 초기화된 임베딩을 사용하여 모델 미세조정을 수행한다.
현재는 구현 예정 단계이다.
"""

from __future__ import annotations

import logging
from typing import Any

from gpt2_ivr.commands.base import Command


class TrainCommand(Command):
    """학습 커맨드.

    재할당된 토크나이저와 임베딩으로 GPT-2 모델을 미세조정한다.
    향후 accelerate 기반 학습 파이프라인이 추가될 예정이다.

    Attributes:
        logger: 로거 인스턴스
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("gpt2_ivr.train")

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """학습 단계를 실행한다.

        현재는 미구현 상태이며 향후 구현 예정이다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            실행 결과 딕셔너리 (status, message)
        """
        self.logger.info("🚧 train 단계는 아직 구현 중입니다.")
        self.logger.info(
            "향후 `src/gpt2_ivr/training/train.py`를 통해 accelerate 기반 학습을 연결합니다."
        )
        return {
            "status": "not_implemented",
            "message": "train 단계 구현이 필요합니다.",
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "train"
        """
        return "train"
