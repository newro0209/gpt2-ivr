"""GPT2-IVR CLI 진입점 모듈.

Tokenizer Model Migration + IVR 파이프라인의 명령줄 인터페이스를 제공한다.
Rich 기반 콘솔 출력 및 로깅을 지원한다.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from pkgutil import iter_modules
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


from gpt2_ivr.commands.base import Command
from gpt2_ivr.constants import LOGS_DIR
from gpt2_ivr.parser import setup_parser

console = Console(stderr=False)


@lru_cache(maxsize=1)
def discover_command_classes() -> tuple[type[Command], ...]:
    """commands 패키지의 Command 서브클래스를 동적으로 탐색한다.

    Returns:
        탐색된 Command 서브클래스 튜플
    """
    import gpt2_ivr.commands as commands_pkg

    package_prefix = f"{commands_pkg.__name__}."
    for module in iter_modules(commands_pkg.__path__, package_prefix):
        if module.name.endswith(".base"):
            continue
        importlib.import_module(module.name)

    command_classes = [
        cls
        for cls in Command.__subclasses__()
        if not inspect.isabstract(cls) and cls.__module__.startswith(package_prefix)
    ]
    command_classes.sort(key=lambda cls: (cls.__module__, cls.__name__))
    return tuple(command_classes)


def _build_command_init_kwargs(args: argparse.Namespace, command_cls: type[Command]) -> dict[str, Any]:
    """Command 생성자 인자를 argparse 네임스페이스로부터 구성한다.

    Args:
        args: 파싱된 커맨드라인 인자
        command_cls: 생성할 Command 클래스

    Returns:
        생성자 키워드 인자 딕셔너리

    Raises:
        ValueError: 필요한 인자가 네임스페이스에 없는 경우
    """
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(command_cls.__init__)
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.name == "console":
            kwargs[param.name] = console
            continue
        if hasattr(args, param.name):
            kwargs[param.name] = getattr(args, param.name)
            continue
        if param.default is inspect._empty:
            raise ValueError(
                f"{command_cls.__name__} 생성에 필요한 인자 '{param.name}'이(가) 파서에 등록되지 않았습니다."
            )
    return kwargs


def setup_logging(log_level: str) -> logging.Logger:
    """로깅을 설정한다.

    Rich 콘솔 핸들러와 파일 핸들러를 모두 설정한다.

    Args:
        log_level: 로깅 레벨 문자열 (DEBUG, INFO, WARNING, ERROR)

    Returns:
        설정된 로거 객체
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Rich 콘솔 핸들러
    console_handler = RichHandler(rich_tracebacks=True, markup=True, console=console, show_time=False)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 파일 핸들러
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"ivr_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    root_logger.addHandler(file_handler)

    root_logger.info("%s에 로그 파일 생성 완료", log_file)
    return root_logger


def print_banner() -> None:
    """시작 배너를 출력한다."""
    from pyfiglet import Figlet

    text = Figlet(font="standard").renderText("IVR").rstrip()
    console.print(Text(text, style="bold cyan"))


def create_command(args: argparse.Namespace) -> Command:
    """커맨드 객체를 생성한다.

    Command 서브클래스 자동 탐색 결과를 기반으로 커맨드를 생성한다.

    Args:
        args: 파싱된 커맨드라인 인자

    Returns:
        생성된 Command 객체

    Raises:
        NotImplementedError: 유효하지 않은 커맨드인 경우
    """
    for command_cls in discover_command_classes():
        try:
            kwargs = _build_command_init_kwargs(args, command_cls)
        except ValueError:
            continue

        command = command_cls(**kwargs)
        if command.get_name() == args.command:
            return command

    raise NotImplementedError(f"'{args.command}'는 유효하지 않은 커맨드입니다.")


def format_time(elapsed: float) -> str:
    """경과 시간을 사람이 읽기 쉬운 형태로 포맷팅한다.

    1초 미만은 밀리초, 1분 미만은 초, 그 이상은 분:초 형식으로 표시한다.

    Args:
        elapsed: 경과 시간 (초 단위)

    Returns:
        포맷팅된 시간 문자열 (예: "500ms", "3.14초", "2분 30.5초")
    """
    if elapsed < 1:
        return f"{elapsed*1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.2f}초"
    minutes, seconds = divmod(elapsed, 60)
    return f"{int(minutes)}분 {seconds:.1f}초"


def format_value(value: Any) -> str:
    """결과 값을 포맷팅한다.

    타입별로 적절한 포맷터를 적용하고, 120자를 초과하면 잘라낸다.

    Args:
        value: 포맷팅할 값 (Any 타입)

    Returns:
        포맷팅된 문자열 (120자 초과 시 "..." 추가)
    """
    formatters = {
        Path: str,
        dict: lambda v: f"dict({len(v)})",
        list: lambda v: f"list({len(v)})",
    }
    formatted = formatters.get(type(value), str)(value)
    return formatted[:117] + "..." if len(formatted) > 120 else formatted


def create_result_panel(command_name: str, elapsed: float, result: dict[str, Any]) -> Panel:
    """실행 결과 테이블을 생성한다.

    Args:
        command_name: 커맨드 이름
        elapsed: 경과 시간 (초 단위)
        result: 실행 결과 딕셔너리

    Returns:
        생성된 Rich Panel 객체
    """
    table = Table(show_header=True, border_style="dim", padding=(0, 1))
    table.add_column("항목", style="bold cyan", width=25)
    table.add_column("값", style="yellow", justify="left")

    table.add_row("⏱️  실행 시간", format_time(elapsed))

    for key, value in result.items():
        formatted_key = key.replace("_", " ").title()
        table.add_row(f"   {formatted_key}", format_value(value))

    return Panel(table, title=f"[bold green]✅ {command_name} 완료[/bold green]", border_style="green", padding=(1, 2))


# Error categorization strategy (Strategy pattern)
_ERROR_CATEGORIES = {
    NotImplementedError: ("미구현 기능", "⚠️", "미구현/미지원 오류"),
    FileNotFoundError: ("파일 없음", "📁", "파일 찾기 실패"),
    ValueError: ("입력값 오류", "⚠️", "입력값 오류"),
}


def handle_error(error: Exception, command: str, elapsed: float, logger: logging.Logger) -> None:
    """에러를 처리하고 출력한다.

    Strategy 패턴을 사용하여 에러 타입별로 적절한 카테고리와 아이콘을 선택한다.

    Args:
        error: 발생한 예외
        command: 실행 중이던 커맨드 이름
        elapsed: 경과 시간 (초 단위)
        logger: 로거 객체
    """
    error_type = type(error).__name__
    category, icon, log_msg = _ERROR_CATEGORIES.get(
        type(error), ("예기치 않은 오류", "❌", "실행 중 예기치 않은 오류 발생")
    )

    # 로깅
    if type(error) in _ERROR_CATEGORIES:
        logger.error("%s %s: %s", command, log_msg, error)
    else:
        logger.exception("%s %s", command, log_msg)

    # Rich 테이블로 에러 정보 구성
    error_table = Table(show_header=False, border_style="dim red", padding=(0, 1))
    error_table.add_column("항목", style="bold red", width=15)
    error_table.add_column("내용", style="white")

    error_table.add_row("카테고리", f"{icon} {category}")
    error_table.add_row("오류 타입", error_type)
    error_table.add_row("메시지", str(error))
    error_table.add_row("경과 시간", format_time(elapsed))

    # Panel로 감싸서 출력
    console.print()
    console.print(
        Panel(error_table, title=f"[bold red]❌ {command} 실행 실패[/bold red]", border_style="red", padding=(1, 2))
    )
    console.print()

    # 도움말 제안
    help_text = Text()
    help_text.append("💡 도움말: ", style="bold yellow")
    help_text.append(f"ivr {command} --help", style="cyan")
    help_text.append(" 명령으로 상세 옵션을 확인하세요", style="dim")
    console.print(help_text)
    console.print()


def main() -> int:
    """CLI 엔트리 포인트.

    파이프라인 명령어를 파싱하고 실행한다. 각 단계별 명령어는
    서브커맨드로 제공되며, Rich 기반 콘솔 출력과 파일 로깅을 지원한다.

    Returns:
        종료 코드 (0: 성공, 1: 오류, 130: 사용자 중단)
    """
    print_banner()
    args = setup_parser(console, discover_command_classes()).parse_args()
    logger = setup_logging(args.log_level)
    start = perf_counter()

    try:
        command = create_command(args)
        command_name = command.get_name()
        logger.info("%s 단계 시작", command_name)
        result = command.execute()
        elapsed = perf_counter() - start

        logger.info("%s 단계 완료 (%.2fs)", command_name, elapsed)
        console.print(create_result_panel(command_name, elapsed, result))
        return 0

    except KeyboardInterrupt:
        logger.warning("사용자 요청으로 실행 중단됨")
        return 130

    except Exception as e:
        handle_error(e, args.command, perf_counter() - start, logger)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
