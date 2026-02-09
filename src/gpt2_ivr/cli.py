"""GPT2-IVR CLI 진입점 모듈.

Tokenizer Model Migration + IVR 파이프라인의 명령줄 인터페이스를 제공한다.
Rich 기반 콘솔 출력 및 로깅을 지원한다.
"""

from __future__ import annotations

import argparse
import logging

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pyfiglet import Figlet
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
    EMBEDDINGS_ROOT,
    REPLACEMENT_CANDIDATES_FILE,
    SELECTION_LOG_FILE,
    TOKENIZER_DISTILLED_UNIGRAM_DIR,
    TOKENIZER_ORIGINAL_DIR,
    TOKENIZER_REMAPPED_DIR,
    TOKEN_FREQUENCY_FILE,
    LOGS_DIR,
)

LOGGER_NAME = "gpt2_ivr.cli"
REMAP_RULES_PATH = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml")

_CONSOLE = Console(stderr=False)


class CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
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
        console = _CONSOLE
        console.print(
            Panel.fit(
                f"[bold red]인자 오류[/bold red]\n{message}\n\n"
                f"[dim]도움말: uv run ivr --help[/dim]",
                title="CLI 입력 오류",
                border_style="red",
            )
        )
        raise SystemExit(2)


def non_negative_int(value: str) -> int:
    """0 이상의 정수 인자를 파싱한다.

    Args:
        value: 파싱할 문자열 값

    Returns:
        파싱된 0 이상의 정수 값

    Raises:
        argparse.ArgumentTypeError: 값이 정수가 아니거나 0보다 작은 경우
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from e

    if parsed < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수만 허용됩니다.")
    return parsed


def positive_int(value: str) -> int:
    """1 이상의 정수 인자를 파싱한다.

    Args:
        value: 파싱할 문자열 값

    Returns:
        파싱된 1 이상의 정수 값

    Raises:
        argparse.ArgumentTypeError: 값이 정수가 아니거나 0 이하인 경우
    """
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("정수를 입력해야 합니다.") from e

    if parsed <= 0:
        raise argparse.ArgumentTypeError("1 이상의 정수만 허용됩니다.")
    return parsed


def main() -> int:
    """CLI 엔트리 포인트.

    파이프라인 명령어를 파싱하고 실행한다. 각 단계별 명령어는
    서브커맨드로 제공되며, Rich 기반 콘솔 출력과 파일 로깅을 지원한다.

    Returns:
        종료 코드 (0: 성공, 1: 오류, 130: 사용자 중단)
    """
    # 1. CLI 파서와 서브커맨드를 정의한다.
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

    subparsers.add_parser(
        "train",
        help="미세조정",
        formatter_class=CliHelpFormatter,
    )

    args = parser.parse_args()

    # 2. 로깅 설정
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()

    root_logger.setLevel(log_level)

    # 2.2. Rich 콘솔 핸들러를 등록한다.
    console_handler = RichHandler(
        rich_tracebacks=True, markup=True, console=_CONSOLE, show_time=False
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 2.3. 파일 핸들러를 등록하여 전체 로그를 기록한다.
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ivr_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
        )
    )
    root_logger.addHandler(file_handler)

    # 2.4. 로그 파일 위치를 안내한다.
    root_logger.info("📝 로그 파일: %s", log_file)

    logger = logging.getLogger(LOGGER_NAME)

    # 3. 인트로 배너를 출력한다.
    console = _CONSOLE
    title = "Tokenizer Model Migration + IVR"
    subtitle = f"실행 명령어: {args.command}"

    figlet = Figlet(font="standard")
    banner = figlet.renderText("GPT2-IVR").rstrip()
    console.print(
        Panel.fit(
            Text(banner, style="bold cyan"),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
        )
    )

    # 4. 명령어를 해석하고 실행한다.
    start = perf_counter()

    try:
        command_name = args.command
        if command_name == "init":
            command = InitCommand(
                model_name=args.model_name,
                tokenizer_dir=args.tokenizer_dir,
                force=args.force,
            )
        elif command_name == "analyze":
            command = AnalyzeCommand(
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
        elif command_name == "distill-tokenizer":
            command = DistillCommand(
                original_tokenizer_dir=args.original_tokenizer_dir,
                distilled_tokenizer_dir=args.distilled_tokenizer_dir,
                corpus_dir=args.corpus_dir,
            )
        elif command_name == "select":
            command = SelectCommand(
                frequency_path=args.frequency_path,
                sequences_path=args.sequences_path,
                output_csv=args.output_csv,
                output_log=args.output_log,
                tokenizer_dir=args.tokenizer_dir,
                max_candidates=args.max_candidates,
                min_token_len=args.min_token_len,
            )
        elif command_name == "remap":
            command = RemapCommand(
                distilled_tokenizer_dir=args.distilled_tokenizer_dir,
                remapped_tokenizer_dir=args.remapped_tokenizer_dir,
                remap_rules_path=args.remap_rules_path,
                replacement_candidates_path=args.replacement_candidates_path,
            )
        elif command_name == "align":
            command = AlignCommand(
                model_name=args.model_name,
                original_tokenizer_dir=args.original_tokenizer_dir,
                remapped_tokenizer_dir=args.remapped_tokenizer_dir,
                remap_rules_path=args.remap_rules_path,
                embeddings_output_dir=args.embeddings_output_dir,
                init_strategy=args.init_strategy,
            )
        elif command_name == "train":
            command = TrainCommand()
        else:
            raise NotImplementedError(f"'{command_name}'는 유효하지 않은 커맨드입니다.")

        resolved_name = command.get_name()
        logger.info("🚀 [%s] 단계를 시작합니다.", resolved_name)
        result = command.execute()
        elapsed = perf_counter() - start
        logger.info("✅ [%s] 단계가 완료되었습니다. (%.2fs)", resolved_name, elapsed)
        table = Table(
            title=f"✅ {resolved_name} 단계 완료",
            show_header=False,
            border_style="green",
        )
        table.add_column("항목", style="bold")
        table.add_column("값")
        table.add_row("실행 시간", f"{elapsed:.2f}초")

        # 4.1. 실행 결과를 테이블로 정리하여 출력한다.
        for key, value in result.items():
            _value_to_format = value
            if isinstance(_value_to_format, Path):
                formatted_value = str(_value_to_format)
            elif isinstance(_value_to_format, dict):
                formatted_value = f"dict({len(_value_to_format)})"
            elif isinstance(_value_to_format, list):
                formatted_value = f"list({len(_value_to_format)})"
            else:
                formatted_value = str(_value_to_format)

            if len(formatted_value) > 120:
                formatted_value = f"{formatted_value[:117]}..."
            table.add_row(str(key), formatted_value)

        _CONSOLE.print(table)
        return 0
    except NotImplementedError as e:
        elapsed = perf_counter() - start
        logger.error("[%s] 미구현/미지원 오류: %s", args.command, e)
        # 4.2. 오류 상황을 Rich 패널로 안내한다.
        _CONSOLE.print(
            Panel.fit(
                f"[bold red]{args.command} 단계 실행 실패[/bold red]\n"
                f"{type(e).__name__}: {e}\n"
                f"[dim]경과 시간: {elapsed:.2f}초[/dim]",
                title="실행 오류",
                border_style="red",
            )
        )
        return 1
    except (FileNotFoundError, ValueError) as e:
        elapsed = perf_counter() - start
        logger.error("[%s] 입력 검증 오류: %s", args.command, e)
        # 4.3. 오류 상황을 Rich 패널로 안내한다.
        _CONSOLE.print(
            Panel.fit(
                f"[bold red]{args.command} 단계 실행 실패[/bold red]\n"
                f"{type(e).__name__}: {e}\n"
                f"[dim]경과 시간: {elapsed:.2f}초[/dim]",
                title="실행 오류",
                border_style="red",
            )
        )
        return 1
    except KeyboardInterrupt:
        logger.warning("⏹️ 사용자 요청으로 실행이 중단되었습니다.")
        return 130
    except Exception as e:
        elapsed = perf_counter() - start
        logger.exception("[%s] 실행 중 예기치 않은 오류가 발생했습니다.", args.command)
        # 4.4. 오류 상황을 Rich 패널로 안내한다.
        _CONSOLE.print(
            Panel.fit(
                f"[bold red]{args.command} 단계 실행 실패[/bold red]\n"
                f"{type(e).__name__}: {e}\n"
                f"[dim]경과 시간: {elapsed:.2f}초[/dim]",
                title="실행 오류",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
