"""학습 커맨드.

재할당 토크나이저와 초기화된 임베딩을 사용하여 모델 미세조정을 수행한다.
현재는 구현 예정 단계이다.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel

from gpt2_ivr.commands.base import Command, SubparsersLike
from gpt2_ivr.parser import CliHelpFormatter

logger = logging.getLogger(__name__)


class TrainCommand(Command):
    """학습 커맨드.

    재할당된 토크나이저와 임베딩으로 GPT-2 모델을 미세조정한다.
    향후 accelerate 기반 학습 파이프라인이 추가될 예정이다.

    Attributes:
        console: Rich 콘솔 인스턴스
    """

    @staticmethod
    def configure_parser(subparsers: SubparsersLike) -> None:
        """서브커맨드 파서를 설정한다.

        Args:
            subparsers: 서브파서 액션 객체
        """
        subparsers.add_parser("train", help="미세조정", formatter_class=CliHelpFormatter)

    def __init__(self, console: Console) -> None:
        self.console = console

    def execute(self) -> dict[str, Any]:
        """학습 단계를 실행한다.

        현재는 미구현 상태이며 향후 구현 예정이다.

        Returns:
            실행 결과 딕셔너리 (status, message)
        """
        message_text = """[yellow]train 단계는 아직 구현 중입니다.[/yellow]

향후 [cyan]src/gpt2_ivr/training/train.py[/cyan]를 통해
accelerate 기반 학습을 연결할 예정입니다."""

        self.console.print()
        self.console.print(Panel(message_text, title="train 단계", border_style="yellow"))
        self.console.print()

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
