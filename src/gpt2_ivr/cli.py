from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from pyfiglet import Figlet
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
    EMBEDDINGS_ROOT,
    REPLACEMENT_CANDIDATES_FILE,
    SELECTION_LOG_FILE,
    TOKENIZER_DISTILLED_UNIGRAM_DIR,
    TOKENIZER_ORIGINAL_DIR,
    TOKENIZER_REMAPPED_DIR,
    TOKEN_FREQUENCY_FILE,
)
from gpt2_ivr.utils.logging_config import get_console, get_logger, setup_logging

LOGGER_NAME = "gpt2_ivr.cli"
REMAP_RULES_PATH = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml")


class CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """CLI 도움말 포맷터."""


class CliArgumentParser(argparse.ArgumentParser):
    """오류 메시지를 Rich 스타일로 출력하는 argparse 파서."""

    def error(self, message: str) -> None:
        console = get_console()
        console.print(
            Panel.fit(
                f"[bold red]인자 오류[/bold red]\n{message}\n\n"
                "[dim]도움말: uv run ivr --help[/dim]",
                title="CLI 입력 오류",
                border_style="red",
            )
        )
        raise SystemExit(2)


def non_negative_int(value: str) -> int:
    """0 이상의 정수 인자를 파싱한다."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from e

    if parsed < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수만 허용됩니다.")
    return parsed


def positive_int(value: str) -> int:
    """1 이상의 정수 인자를 파싱한다."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from e

    if parsed <= 0:
        raise argparse.ArgumentTypeError("1 이상의 정수만 허용됩니다.")
    return parsed


def build_banner(text: str, font: str = "standard") -> str:
    """배너 문자열을 생성한다."""
    figlet = Figlet(font=font)
    return figlet.renderText(text)


def build_parser() -> argparse.ArgumentParser:
    """IVR CLI 파서를 생성한다."""
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
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        default=False,
        help="artifacts/logs 파일 로그 기록 비활성화",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        default=False,
        help="시작 ASCII 배너 출력 비활성화",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    # init 서브커맨드
    init_parser = subparsers.add_parser(
        "init",
        help="모델 및 토크나이저 초기화",
        formatter_class=CliHelpFormatter,
    )
    init_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="Hugging Face Hub 모델 이름",
    )
    init_parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=TOKENIZER_ORIGINAL_DIR,
        help="토크나이저 저장 디렉토리",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="기존 파일이 있어도 다시 다운로드",
    )

    # analyze 서브커맨드
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="BPE 토큰 시퀀스 분석",
        formatter_class=CliHelpFormatter,
    )
    analyze_parser.add_argument(
        "--input-dir",
        type=Path,
        default=CORPORA_CLEANED_DIR,
        help="코퍼스 입력 디렉토리",
    )
    analyze_parser.add_argument(
        "--output-sequences",
        type=Path,
        default=BPE_TOKEN_ID_SEQUENCES_FILE,
        help="BPE 토큰 시퀀스 출력 경로",
    )
    analyze_parser.add_argument(
        "--output-frequency",
        type=Path,
        default=TOKEN_FREQUENCY_FILE,
        help="토큰 빈도 parquet 출력 경로",
    )
    analyze_parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=TOKENIZER_ORIGINAL_DIR,
        help="원본 토크나이저 디렉토리",
    )
    analyze_parser.add_argument(
        "--workers",
        type=non_negative_int,
        default=0,
        help="스레드 워커 수 (0이면 CPU - 1)",
    )
    analyze_parser.add_argument(
        "--chunk-size",
        type=non_negative_int,
        default=0,
        help="스레드 청크 크기(0이면 자동 설정)",
    )
    analyze_parser.add_argument(
        "--max-texts",
        type=non_negative_int,
        default=0,
        help="처리할 최대 텍스트 수 (0이면 전체)",
    )
    analyze_parser.add_argument(
        "--text-key",
        default="text",
        help="json/jsonl 텍스트 키",
    )
    analyze_parser.add_argument(
        "--encoding",
        default="utf-8",
        help="입력 파일 인코딩",
    )

    # distill-tokenizer 서브커맨드
    distill_parser = subparsers.add_parser(
        "distill-tokenizer",
        help="BPE -> Unigram distillation",
        formatter_class=CliHelpFormatter,
    )
    distill_parser.add_argument(
        "--original-tokenizer-dir",
        type=Path,
        default=TOKENIZER_ORIGINAL_DIR,
        help="원본 토크나이저 디렉토리",
    )
    distill_parser.add_argument(
        "--distilled-tokenizer-dir",
        type=Path,
        default=TOKENIZER_DISTILLED_UNIGRAM_DIR,
        help="증류된 토크나이저 저장 디렉토리",
    )
    distill_parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPORA_CLEANED_DIR,
        help="학습 코퍼스 디렉토리",
    )

    # select 서브커맨드
    select_parser = subparsers.add_parser(
        "select",
        help="IVR 대상 토큰 선정",
        formatter_class=CliHelpFormatter,
    )
    select_parser.add_argument(
        "--frequency-path",
        type=Path,
        default=TOKEN_FREQUENCY_FILE,
        help="토큰 빈도 parquet 파일 경로",
    )
    select_parser.add_argument(
        "--sequences-path",
        type=Path,
        default=BPE_TOKEN_ID_SEQUENCES_FILE,
        help="BPE 토큰 시퀀스 파일 경로",
    )
    select_parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPLACEMENT_CANDIDATES_FILE,
        help="교체 후보 CSV 저장 경로",
    )
    select_parser.add_argument(
        "--output-log",
        type=Path,
        default=SELECTION_LOG_FILE,
        help="선정 로그 저장 경로",
    )
    select_parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=TOKENIZER_ORIGINAL_DIR,
        help="원본 토크나이저 디렉토리",
    )
    select_parser.add_argument(
        "--max-candidates",
        type=positive_int,
        default=1000,
        help="최대 후보 개수",
    )
    select_parser.add_argument(
        "--min-token-len",
        type=positive_int,
        default=2,
        help="보호 토큰 최소 길이",
    )

    # remap 서브커맨드
    remap_parser = subparsers.add_parser(
        "remap",
        help="토큰 재할당 규칙 적용",
        formatter_class=CliHelpFormatter,
    )
    remap_parser.add_argument(
        "--distilled-tokenizer-dir",
        type=Path,
        default=TOKENIZER_DISTILLED_UNIGRAM_DIR,
        help="증류된 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remapped-tokenizer-dir",
        type=Path,
        default=TOKENIZER_REMAPPED_DIR,
        help="재할당 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remap-rules-path",
        type=Path,
        default=REMAP_RULES_PATH,
        help="재할당 규칙 파일 경로",
    )
    remap_parser.add_argument(
        "--replacement-candidates-path",
        type=Path,
        default=REPLACEMENT_CANDIDATES_FILE,
        help="교체 후보 CSV 경로",
    )

    # align 서브커맨드
    align_parser = subparsers.add_parser(
        "align",
        help="임베딩 재정렬",
        formatter_class=CliHelpFormatter,
    )
    align_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="GPT-2 모델 이름",
    )
    align_parser.add_argument(
        "--original-tokenizer-dir",
        type=Path,
        default=TOKENIZER_ORIGINAL_DIR,
        help="원본 토크나이저 디렉토리",
    )
    align_parser.add_argument(
        "--remapped-tokenizer-dir",
        type=Path,
        default=TOKENIZER_REMAPPED_DIR,
        help="재할당 토크나이저 디렉토리",
    )
    align_parser.add_argument(
        "--remap-rules-path",
        type=Path,
        default=REMAP_RULES_PATH,
        help="재할당 규칙 파일 경로",
    )
    align_parser.add_argument(
        "--embeddings-output-dir",
        type=Path,
        default=EMBEDDINGS_ROOT,
        help="임베딩 출력 디렉토리",
    )
    align_parser.add_argument(
        "--init-strategy",
        default="mean",
        choices=["mean", "random", "zeros"],
        help="신규 토큰 임베딩 초기화 전략",
    )

    # train 서브커맨드 (현재 stub)
    subparsers.add_parser(
        "train",
        help="미세조정",
        formatter_class=CliHelpFormatter,
    )

    return parser


def format_result_value(value: Any) -> str:
    """결과 값 출력 문자열을 정규화한다."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return f"dict({len(value)})"
    if isinstance(value, list):
        return f"list({len(value)})"

    rendered = str(value)
    if len(rendered) > 120:
        return f"{rendered[:117]}..."
    return rendered


def render_intro(command_name: str, show_banner: bool) -> None:
    """실행 시작 정보를 출력한다."""
    console = get_console()
    title = "Tokenizer Model Migration + IVR"
    subtitle = f"실행 명령어: {command_name}"

    if show_banner:
        banner = build_banner("GPT2-IVR").rstrip()
        console.print(
            Panel.fit(
                Text(banner, style="bold cyan"),
                title=title,
                subtitle=subtitle,
                border_style="cyan",
            )
        )
        return

    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]\n{subtitle}",
            border_style="cyan",
        )
    )


def render_result_summary(
    command_name: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    """커맨드 실행 결과를 테이블로 출력한다."""
    table = Table(
        title=f"✅ {command_name} 단계 완료",
        show_header=False,
        border_style="green",
    )
    table.add_column("항목", style="bold")
    table.add_column("값")
    table.add_row("실행 시간", f"{elapsed_seconds:.2f}초")

    for key, value in result.items():
        table.add_row(str(key), format_result_value(value))

    get_console().print(table)


def render_error_panel(
    command_name: str,
    error: Exception,
    elapsed_seconds: float,
) -> None:
    """실행 오류를 패널로 출력한다."""
    get_console().print(
        Panel.fit(
            f"[bold red]{command_name} 단계 실행 실패[/bold red]\n"
            f"{type(error).__name__}: {error}\n"
            f"[dim]경과 시간: {elapsed_seconds:.2f}초[/dim]",
            title="실행 오류",
            border_style="red",
        )
    )


def _create_init_command(args: argparse.Namespace) -> InitCommand:
    """init 커맨드를 생성한다."""
    return InitCommand(
        model_name=args.model_name,
        tokenizer_dir=args.tokenizer_dir,
        force=args.force,
    )


def _create_analyze_command(args: argparse.Namespace) -> AnalyzeCommand:
    """analyze 커맨드를 생성한다."""
    return AnalyzeCommand(
        input_dir=args.input_dir,
        output_sequences=args.output_sequences,
        output_frequency=args.output_frequency,
        tokenizer_dir=args.tokenizer_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_texts=args.max_texts,
        text_key=args.text_key,
        encoding=args.encoding,
    )


def _create_distill_command(args: argparse.Namespace) -> DistillCommand:
    """distill-tokenizer 커맨드를 생성한다."""
    return DistillCommand(
        original_tokenizer_dir=args.original_tokenizer_dir,
        distilled_tokenizer_dir=args.distilled_tokenizer_dir,
        corpus_dir=args.corpus_dir,
    )


def _create_select_command(args: argparse.Namespace) -> SelectCommand:
    """select 커맨드를 생성한다."""
    return SelectCommand(
        frequency_path=args.frequency_path,
        sequences_path=args.sequences_path,
        output_csv=args.output_csv,
        output_log=args.output_log,
        tokenizer_dir=args.tokenizer_dir,
        max_candidates=args.max_candidates,
        min_token_len=args.min_token_len,
    )


def _create_remap_command(args: argparse.Namespace) -> RemapCommand:
    """remap 커맨드를 생성한다."""
    return RemapCommand(
        distilled_tokenizer_dir=args.distilled_tokenizer_dir,
        remapped_tokenizer_dir=args.remapped_tokenizer_dir,
        remap_rules_path=args.remap_rules_path,
        replacement_candidates_path=args.replacement_candidates_path,
    )


def _create_align_command(args: argparse.Namespace) -> AlignCommand:
    """align 커맨드를 생성한다."""
    return AlignCommand(
        model_name=args.model_name,
        original_tokenizer_dir=args.original_tokenizer_dir,
        remapped_tokenizer_dir=args.remapped_tokenizer_dir,
        remap_rules_path=args.remap_rules_path,
        embeddings_output_dir=args.embeddings_output_dir,
        init_strategy=args.init_strategy,
    )


def _create_train_command(args: argparse.Namespace) -> TrainCommand:
    """train 커맨드를 생성한다. (현재 stub)"""
    # TODO: train 커맨드에 CLI 옵션이 추가되면 args를 사용하여 파라미터 전달
    return TrainCommand()


# 서브커맨드 팩토리 매핑
CommandFactory = Callable[[argparse.Namespace], Command]

COMMAND_FACTORY_MAP: dict[str, CommandFactory] = {
    "init": _create_init_command,
    "analyze": _create_analyze_command,
    "distill-tokenizer": _create_distill_command,
    "select": _create_select_command,
    "remap": _create_remap_command,
    "align": _create_align_command,
    "train": _create_train_command,
}


def create_command(command_name: str, args: argparse.Namespace) -> Command:
    """커맨드 이름에 해당하는 Command 객체를 생성한다."""
    factory = COMMAND_FACTORY_MAP.get(command_name)
    if factory is not None:
        return factory(args)

    raise NotImplementedError(f"'{command_name}'는 유효하지 않은 커맨드입니다.")


def dispatch(
    command_name: str, args: argparse.Namespace, logger: logging.Logger
) -> int:
    """서브커맨드를 실행한다."""
    start = perf_counter()

    try:
        command = create_command(command_name, args)
        resolved_name = command.get_name()
        logger.info("🚀 [%s] 단계를 시작합니다.", resolved_name)
        result = command.execute()
        elapsed = perf_counter() - start
        logger.info("✅ [%s] 단계가 완료되었습니다. (%.2fs)", resolved_name, elapsed)
        render_result_summary(resolved_name, result, elapsed)
        return 0
    except NotImplementedError as e:
        elapsed = perf_counter() - start
        logger.error("[%s] 미구현/미지원 오류: %s", command_name, e)
        render_error_panel(command_name, e, elapsed)
        return 1
    except (FileNotFoundError, ValueError) as e:
        elapsed = perf_counter() - start
        logger.error("[%s] 입력 검증 오류: %s", command_name, e)
        render_error_panel(command_name, e, elapsed)
        return 1
    except KeyboardInterrupt:
        logger.warning("⏹️ 사용자 요청으로 실행이 중단되었습니다.")
        return 130
    except Exception as e:
        elapsed = perf_counter() - start
        logger.exception("[%s] 실행 중 예기치 않은 오류가 발생했습니다.", command_name)
        render_error_panel(command_name, e, elapsed)
        return 1


def main() -> int:
    """CLI 엔트리 포인트."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level, log_to_file=not args.no_log_file)
    logger = get_logger(LOGGER_NAME)

    render_intro(args.command, show_banner=not args.no_banner)
    logger.info("Tokenizer Model Migration + IVR 파이프라인")
    logger.info("BPE -> Unigram 토크나이저 교체 후 IVR를 수행합니다.")
    return dispatch(args.command, args, logger)


if __name__ == "__main__":
    raise SystemExit(main())
