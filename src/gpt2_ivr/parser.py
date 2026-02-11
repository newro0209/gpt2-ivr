"""CLI 인자 파서 설정 모듈.

argparse 기반 CLI 파서와 서브커맨드를 정의한다.
검증 함수와 서브파서 구성을 담당한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from gpt2_ivr.commands.base import Command


class CliHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """CLI 도움말 포맷터.

    ArgumentDefaultsHelpFormatter와 RawTextHelpFormatter를 결합하여
    기본값 표시와 원시 텍스트 포맷을 동시에 지원한다.
    """


class CliArgumentParser(argparse.ArgumentParser):
    """오류 메시지를 Rich 스타일로 출력하는 argparse 파서.

    인자 파싱 오류 발생 시 Rich Panel로 오류를 표시하여
    사용자 경험을 개선한다.

    Attributes:
        console: Rich 콘솔 인스턴스
    """

    def __init__(self, console: Console | None = None, **kwargs: Any) -> None:
        self.console = console or Console()
        super().__init__(**kwargs)

    def error(self, message: str) -> None:
        """인자 파싱 오류를 Rich 패널로 출력한다.

        Args:
            message: 오류 메시지
        """
        self.console.print(
            Panel.fit(
                f"[bold red]인자 오류[/bold red]\n{message}\n\n[dim]도움말: uv run ivr --help[/dim]",
                title="CLI 입력 오류",
                border_style="red",
            )
        )
        raise SystemExit(2)


def validate_int(value: str, minimum: int = 0) -> int:
    """정수 값을 검증한다.

    Args:
        value: 파싱할 문자열 값
        minimum: 허용되는 최소값 (기본값: 0)

    Returns:
        파싱된 정수 값

    Raises:
        argparse.ArgumentTypeError: 값이 정수가 아니거나 최소값보다 작은 경우
    """
    try:
        if (parsed := int(value)) < minimum:
            raise argparse.ArgumentTypeError(f"{minimum} 이상의 정수만 허용됩니다.")
        return parsed
    except ValueError as e:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from e


def non_negative_int(value: str) -> int:
    """0 이상의 정수를 검증한다."""
    return validate_int(value, minimum=0)


def positive_int(value: str) -> int:
    """1 이상의 정수를 검증한다."""
    return validate_int(value, minimum=1)


def setup_parser(console: Console, commands: Iterable[type[Command]]) -> argparse.ArgumentParser:
    """CLI 파서를 설정한다.

    각 Command 서브클래스의 configure_parser()를 호출하여 서브커맨드를 등록한다.

    Args:
        console: Rich 콘솔 인스턴스 (오류 출력용)
        commands: Command 서브클래스 이터러블

    Returns:
        설정된 ArgumentParser 객체
    """
    parser = CliArgumentParser(
        console,
        prog="ivr",
        description="Tokenizer Model Migration + IVR 파이프라인 CLI",
        formatter_class=CliHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="콘솔 로깅 레벨",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for cmd_cls in commands:
        cmd_cls.configure_parser(subparsers)

    return parser
