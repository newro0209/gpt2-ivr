"""GPT2-IVR CLI 진입점 모듈.

Tokenizer Model Migration + IVR 파이프라인의 명령줄 인터페이스를 제공한다.
Rich 기반 콘솔 출력 및 로깅을 지원한다.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpt2_ivr.commands import (
    AlignCommand,
    AnalyzeCommand,
    Command,
    DistillCommand,
    InitCommand,
    RemapCommand,
    SelectCommand,
    TrainCommand,
)
from gpt2_ivr.constants import (
    BPE_TOKEN_ID_SEQUENCES_FILE,
    CORPORA_CLEANED_DIR,
    CORPORA_RAW_DIR,
    EMBEDDINGS_ROOT,
    LOGS_DIR,
    REPLACEMENT_CANDIDATES_FILE,
    SELECTION_LOG_FILE,
    TOKENIZER_DISTILLED_UNIGRAM_DIR,
    TOKENIZER_ORIGINAL_DIR,
    TOKENIZER_REMAPPED_DIR,
    TOKEN_FREQUENCY_FILE,
)

LOGGER_NAME = "gpt2_ivr.cli"
REMAP_RULES_PATH = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml")
CONSOLE = Console(stderr=False)


# Command registry for Factory pattern
_COMMAND_REGISTRY: dict[str, Callable[[argparse.Namespace], Command]] = {}


def register_command(name: str) -> Callable:
    """커맨드 팩토리 함수를 레지스트리에 등록하는 데코레이터.

    Args:
        name: 커맨드 이름 (CLI 서브커맨드 이름)

    Returns:
        데코레이터 함수
    """
    def decorator(factory: Callable[[argparse.Namespace], Command]) -> Callable:
        _COMMAND_REGISTRY[name] = factory
        return factory
    return decorator


@dataclass
class ArgConfig:
    """공통 인자 설정을 담는 데이터클래스."""
    flag: str
    type: type = str
    default: Any = None
    help: str = ""
    action: str | None = None
    choices: list[str] | None = None


# 공통 인자 설정 (데이터 기반)
COMMON_ARG_CONFIGS = {
    "tokenizer-dir": ArgConfig("--tokenizer-dir", Path, TOKENIZER_ORIGINAL_DIR, "원본 토크나이저 디렉토리"),
    "original-tokenizer-dir": ArgConfig("--original-tokenizer-dir", Path, TOKENIZER_ORIGINAL_DIR, "원본 토크나이저 디렉토리"),
    "distilled-tokenizer-dir": ArgConfig("--distilled-tokenizer-dir", Path, TOKENIZER_DISTILLED_UNIGRAM_DIR, "증류된 토크나이저 디렉토리"),
    "remapped-tokenizer-dir": ArgConfig("--remapped-tokenizer-dir", Path, TOKENIZER_REMAPPED_DIR, "재할당 토크나이저 디렉토리"),
    "remap-rules-path": ArgConfig("--remap-rules-path", Path, REMAP_RULES_PATH, "재할당 규칙 파일 경로"),
}


class CliHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """CLI 도움말 포맷터.

    ArgumentDefaultsHelpFormatter와 RawTextHelpFormatter를 결합하여
    기본값 표시와 원시 텍스트 포맷을 동시에 지원한다.
    """


class CliArgumentParser(argparse.ArgumentParser):
    """오류 메시지를 Rich 스타일로 출력하는 argparse 파서.

    인자 파싱 오류 발생 시 Rich Panel로 오류를 표시하여
    사용자 경험을 개선한다.
    """

    def error(self, message: str) -> None:
        """인자 파싱 오류를 Rich 패널로 출력한다.

        Args:
            message: 오류 메시지
        """
        CONSOLE.print(
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


# Partial application으로 중복 제거
non_negative_int = partial(validate_int, minimum=0)
positive_int = partial(validate_int, minimum=1)


def add_common_args(parser: argparse.ArgumentParser, *args: str) -> None:
    """공통 인자를 파서에 추가한다.

    COMMON_ARG_CONFIGS에서 설정을 조회하여 데이터 기반으로 인자를 추가한다.

    Args:
        parser: 인자를 추가할 파서
        *args: 추가할 인자 이름들 (COMMON_ARG_CONFIGS의 키)
    """
    for arg in args:
        if config := COMMON_ARG_CONFIGS.get(arg):
            kwargs = {"type": config.type, "default": config.default, "help": config.help}
            if config.action:
                kwargs["action"] = config.action
            if config.choices:
                kwargs["choices"] = config.choices
            parser.add_argument(config.flag, **kwargs)


def setup_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """모든 서브커맨드 파서를 설정한다.

    Args:
        subparsers: 서브파서 액션 객체
    """
    # init
    init_parser = subparsers.add_parser("init", help="모델 및 토크나이저 초기화", formatter_class=CliHelpFormatter)
    init_parser.add_argument("--model-name", default="openai-community/gpt2", help="Hugging Face Hub 모델 이름")
    init_parser.add_argument(
        "--tokenizer-dir", type=Path, default=TOKENIZER_ORIGINAL_DIR, help="토크나이저 저장 디렉토리"
    )
    init_parser.add_argument("--force", action="store_true", help="기존 파일이 있어도 다시 다운로드")
    init_parser.add_argument(
        "--raw-corpora-dir",
        type=Path,
        default=CORPORA_RAW_DIR,
        help="raw 코퍼스가 위치한 디렉토리",
    )
    init_parser.add_argument(
        "--cleaned-corpora-dir",
        type=Path,
        default=CORPORA_CLEANED_DIR,
        help="정제된 코퍼스를 저장할 디렉토리",
    )
    init_parser.add_argument("--text-key", default="text", help="JSON/JSONL 파일에서 텍스트를 읽어올 키")
    init_parser.add_argument("--encoding", default="utf-8", help="입력 코퍼스 파일 인코딩")
    init_parser.add_argument(
        "--normalize-force",
        action="store_true",
        help="이미 정제본이 있어도 raw 파일을 다시 변환합니다",
    )

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="BPE 토큰 시퀀스 분석", formatter_class=CliHelpFormatter)
    analyze_parser.add_argument("--input-dir", type=Path, default=CORPORA_CLEANED_DIR, help="코퍼스 입력 디렉토리")
    analyze_parser.add_argument(
        "--output-sequences", type=Path, default=BPE_TOKEN_ID_SEQUENCES_FILE, help="BPE 토큰 시퀀스 출력 경로"
    )
    analyze_parser.add_argument(
        "--output-frequency", type=Path, default=TOKEN_FREQUENCY_FILE, help="토큰 빈도 parquet 출력 경로"
    )
    add_common_args(analyze_parser, "tokenizer-dir")
    analyze_parser.add_argument("--workers", type=non_negative_int, default=0, help="스레드 워커 수 (0이면 CPU - 1)")
    analyze_parser.add_argument(
        "--chunk-size", type=non_negative_int, default=0, help="스레드 청크 크기(0이면 자동 설정)"
    )
    analyze_parser.add_argument(
        "--max-texts", type=non_negative_int, default=0, help="처리할 최대 텍스트 수 (0이면 전체)"
    )
    analyze_parser.add_argument("--encoding", default="utf-8", help="입력 파일 인코딩")

    # distill-tokenizer
    distill_parser = subparsers.add_parser(
        "distill-tokenizer", help="BPE -> Unigram distillation", formatter_class=CliHelpFormatter
    )
    add_common_args(distill_parser, "original-tokenizer-dir", "distilled-tokenizer-dir")
    distill_parser.add_argument("--corpus-dir", type=Path, default=CORPORA_CLEANED_DIR, help="학습 코퍼스 디렉토리")

    # select
    select_parser = subparsers.add_parser("select", help="IVR 대상 토큰 선정", formatter_class=CliHelpFormatter)
    select_parser.add_argument(
        "--frequency-path", type=Path, default=TOKEN_FREQUENCY_FILE, help="토큰 빈도 parquet 파일 경로"
    )
    select_parser.add_argument(
        "--sequences-path", type=Path, default=BPE_TOKEN_ID_SEQUENCES_FILE, help="BPE 토큰 시퀀스 파일 경로"
    )
    select_parser.add_argument(
        "--output-csv", type=Path, default=REPLACEMENT_CANDIDATES_FILE, help="교체 후보 CSV 저장 경로"
    )
    select_parser.add_argument("--output-log", type=Path, default=SELECTION_LOG_FILE, help="선정 로그 저장 경로")
    add_common_args(select_parser, "tokenizer-dir")
    select_parser.add_argument("--max-candidates", type=positive_int, default=1000, help="최대 후보 개수")
    select_parser.add_argument("--min-token-len", type=positive_int, default=2, help="보호 토큰 최소 길이")

    # remap
    remap_parser = subparsers.add_parser("remap", help="토큰 재할당 규칙 적용", formatter_class=CliHelpFormatter)
    add_common_args(remap_parser, "distilled-tokenizer-dir", "remapped-tokenizer-dir", "remap-rules-path")
    remap_parser.add_argument(
        "--replacement-candidates-path", type=Path, default=REPLACEMENT_CANDIDATES_FILE, help="교체 후보 CSV 경로"
    )

    # align
    align_parser = subparsers.add_parser("align", help="임베딩 재정렬", formatter_class=CliHelpFormatter)
    align_parser.add_argument("--model-name", default="openai-community/gpt2", help="GPT-2 모델 이름")
    add_common_args(align_parser, "original-tokenizer-dir", "remapped-tokenizer-dir", "remap-rules-path")
    align_parser.add_argument(
        "--embeddings-output-dir", type=Path, default=EMBEDDINGS_ROOT, help="임베딩 출력 디렉토리"
    )
    align_parser.add_argument(
        "--init-strategy", default="mean", choices=["mean", "random", "zeros"], help="신규 토큰 임베딩 초기화 전략"
    )

    # train
    subparsers.add_parser("train", help="미세조정", formatter_class=CliHelpFormatter)


def setup_parser() -> argparse.ArgumentParser:
    """CLI 파서를 설정한다.

    Returns:
        설정된 ArgumentParser 객체
    """
    parser = CliArgumentParser(
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
    setup_subparsers(subparsers)

    return parser


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
    console_handler = RichHandler(rich_tracebacks=True, markup=True, console=CONSOLE, show_time=False)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 파일 핸들러
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"ivr_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    root_logger.addHandler(file_handler)

    root_logger.info("%s에 로그 파일 생성 완료", log_file)
    return logging.getLogger(LOGGER_NAME)


@lru_cache(maxsize=1)
def _get_banner() -> str:
    """배너 텍스트를 캐싱하여 반환한다.

    pyfiglet을 사용하여 ASCII 아트 배너를 생성하고, LRU 캐시로 재사용한다.

    Returns:
        생성된 배너 텍스트
    """
    from pyfiglet import Figlet
    return Figlet(font="standard").renderText("IVR").rstrip()


def print_banner() -> None:
    """시작 배너를 출력한다."""
    CONSOLE.print(Text(_get_banner(), style="bold cyan"))


# Command factory functions (Registry pattern)
@register_command("init")
def _create_init_command(a: argparse.Namespace) -> InitCommand:
    """InitCommand 팩토리 함수."""
    return InitCommand(
        a.model_name, a.tokenizer_dir, a.force, a.raw_corpora_dir,
        a.cleaned_corpora_dir, a.text_key, a.encoding, a.normalize_force
    )


@register_command("analyze")
def _create_analyze_command(a: argparse.Namespace) -> AnalyzeCommand:
    """AnalyzeCommand 팩토리 함수."""
    return AnalyzeCommand(
        a.input_dir, a.output_sequences, a.output_frequency,
        a.tokenizer_dir, a.workers, a.chunk_size, a.max_texts, a.encoding
    )


@register_command("distill-tokenizer")
def _create_distill_command(a: argparse.Namespace) -> DistillCommand:
    """DistillCommand 팩토리 함수."""
    return DistillCommand(a.original_tokenizer_dir, a.distilled_tokenizer_dir, a.corpus_dir)


@register_command("select")
def _create_select_command(a: argparse.Namespace) -> SelectCommand:
    """SelectCommand 팩토리 함수."""
    return SelectCommand(
        a.frequency_path, a.sequences_path, a.output_csv, a.output_log,
        a.tokenizer_dir, a.max_candidates, a.min_token_len
    )


@register_command("remap")
def _create_remap_command(a: argparse.Namespace) -> RemapCommand:
    """RemapCommand 팩토리 함수."""
    return RemapCommand(
        a.distilled_tokenizer_dir, a.remapped_tokenizer_dir,
        a.remap_rules_path, a.replacement_candidates_path
    )


@register_command("align")
def _create_align_command(a: argparse.Namespace) -> AlignCommand:
    """AlignCommand 팩토리 함수."""
    return AlignCommand(
        a.model_name, a.original_tokenizer_dir, a.remapped_tokenizer_dir,
        a.remap_rules_path, a.embeddings_output_dir, a.init_strategy
    )


@register_command("train")
def _create_train_command(a: argparse.Namespace) -> TrainCommand:
    """TrainCommand 팩토리 함수."""
    return TrainCommand()


def create_command(args: argparse.Namespace) -> Command:
    """커맨드 객체를 생성한다.

    Factory Registry 패턴을 사용하여 커맨드 이름에 해당하는 팩토리 함수를 조회하고 실행한다.

    Args:
        args: 파싱된 커맨드라인 인자

    Returns:
        생성된 Command 객체

    Raises:
        NotImplementedError: 유효하지 않은 커맨드인 경우
    """
    if factory := _COMMAND_REGISTRY.get(args.command):
        return factory(args)
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


def create_result_table(command_name: str, elapsed: float, result: dict[str, Any]) -> Panel:
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

    return Panel(
        table,
        title=f"[bold green]✅ {command_name} 완료[/bold green]",
        border_style="green",
        padding=(1, 2)
    )


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
    category, icon, log_msg = _ERROR_CATEGORIES.get(type(error), ("예기치 않은 오류", "❌", "실행 중 예기치 않은 오류 발생"))

    # 로깅
    if type(error) in _ERROR_CATEGORIES:
        logger.error("[%s] %s: %s", command, log_msg, error)
    else:
        logger.exception("[%s] %s", command, log_msg)

    # Rich 테이블로 에러 정보 구성
    error_table = Table(show_header=False, border_style="dim red", padding=(0, 1))
    error_table.add_column("항목", style="bold red", width=15)
    error_table.add_column("내용", style="white")

    error_table.add_row("카테고리", f"{icon} {category}")
    error_table.add_row("오류 타입", error_type)
    error_table.add_row("메시지", str(error))
    error_table.add_row("경과 시간", format_time(elapsed))

    # Panel로 감싸서 출력
    CONSOLE.print()
    CONSOLE.print(
        Panel(error_table, title=f"[bold red]❌ {command} 실행 실패[/bold red]",
              border_style="red", padding=(1, 2))
    )
    CONSOLE.print()

    # 도움말 제안
    help_text = Text()
    help_text.append("💡 도움말: ", style="bold yellow")
    help_text.append(f"ivr {command} --help", style="cyan")
    help_text.append(" 명령으로 상세 옵션을 확인하세요", style="dim")
    CONSOLE.print(help_text)
    CONSOLE.print()


def main() -> int:
    """CLI 엔트리 포인트.

    파이프라인 명령어를 파싱하고 실행한다. 각 단계별 명령어는
    서브커맨드로 제공되며, Rich 기반 콘솔 출력과 파일 로깅을 지원한다.

    Returns:
        종료 코드 (0: 성공, 1: 오류, 130: 사용자 중단)
    """
    print_banner()
    args = setup_parser().parse_args()
    logger = setup_logging(args.log_level)
    start = perf_counter()

    try:
        command = create_command(args)
        command_name = command.get_name()
        logger.info("[%s] 단계 시작", command_name)
        result = command.execute()
        elapsed = perf_counter() - start

        logger.info("[%s] 단계 완료 (%.2fs)", command_name, elapsed)
        CONSOLE.print(create_result_table(command_name, elapsed, result))
        return 0

    except KeyboardInterrupt:
        logger.warning("사용자 요청으로 실행 중단됨")
        return 130

    except Exception as e:
        handle_error(e, args.command, perf_counter() - start, logger)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
