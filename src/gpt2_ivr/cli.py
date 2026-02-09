from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pyfiglet import Figlet

from gpt2_ivr.commands import (
    AnalyzeCommand,
    Command,
    DistillCommand,
    InitCommand,
    SelectCommand,
    RemapCommand,
    AlignCommand,
    TrainCommand,
)
from gpt2_ivr.constants import (
    BPE_TOKEN_ID_SEQUENCES_FILE,
    CORPORA_CLEANED_DIR,
    REPLACEMENT_CANDIDATES_FILE,
    SELECTION_LOG_FILE,
    TOKENIZER_DISTILLED_UNIGRAM_DIR,
    TOKENIZER_ORIGINAL_DIR,
    TOKENIZER_REMAPPED_DIR,
    TOKEN_FREQUENCY_FILE,
)
from gpt2_ivr.utils.logging_config import get_logger, setup_logging


def build_banner(text: str, font: str = "standard") -> str:
    """배너 문자열을 생성한다."""
    figlet = Figlet(font=font)
    return figlet.renderText(text)


def build_parser() -> argparse.ArgumentParser:
    """IVR CLI 파서를 생성한다."""
    parser = argparse.ArgumentParser(
        prog="ivr",
        description="Tokenizer Model Migration + IVR 파이프라인 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init 서브커맨드
    init_parser = subparsers.add_parser(
        "init", help="모델 및 토크나이저 초기화 (다운로드)"
    )
    init_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="Hugging Face Hub 모델 이름",
    )
    init_parser.add_argument(
        "--tokenizer-dir",
        default=str(TOKENIZER_ORIGINAL_DIR),
        help="토크나이저 저장 디렉토리",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="기존 파일이 있어도 다시 다운로드",
    )

    # analyze 서브커맨드
    analyze_parser = subparsers.add_parser("analyze", help="BPE 토큰 시퀀스 분석")
    analyze_parser.add_argument(
        "--input-dir",
        default=str(CORPORA_CLEANED_DIR),
        help="코퍼스 입력 디렉토리",
    )
    analyze_parser.add_argument(
        "--output-sequences",
        default=str(BPE_TOKEN_ID_SEQUENCES_FILE),
        help="BPE 토큰 시퀀스 출력 경로",
    )
    analyze_parser.add_argument(
        "--output-frequency",
        default=str(TOKEN_FREQUENCY_FILE),
        help="토큰 빈도 parquet 출력 경로",
    )
    analyze_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="GPT-2 모델 이름",
    )
    analyze_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="스레드 워커 수 (0이면 CPU * 2)",
    )
    analyze_parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="스레드 청크 크기",
    )
    analyze_parser.add_argument(
        "--max-texts",
        type=int,
        default=0,
        help="처리할 최대 텍스트 수 (0이면 전체)",
    )

    # distill-tokenizer 서브커맨드
    distill_parser = subparsers.add_parser(
        "distill-tokenizer", help="BPE -> Unigram distillation"
    )
    distill_parser.add_argument(
        "--original-tokenizer-dir",
        default=str(TOKENIZER_ORIGINAL_DIR),
        help="원본 토크나이저 디렉토리",
    )
    distill_parser.add_argument(
        "--distilled-tokenizer-dir",
        default=str(TOKENIZER_DISTILLED_UNIGRAM_DIR),
        help="증류된 토크나이저 저장 디렉토리",
    )
    distill_parser.add_argument(
        "--corpus-dir",
        default=str(CORPORA_CLEANED_DIR),
        help="학습 코퍼스 디렉토리",
    )
    distill_parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="어휘 크기",
    )
    distill_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="모델 이름",
    )

    # select 서브커맨드
    select_parser = subparsers.add_parser("select", help="IVR 대상 토큰 선정")
    select_parser.add_argument(
        "--frequency-path",
        default=str(TOKEN_FREQUENCY_FILE),
        help="토큰 빈도 parquet 파일 경로",
    )
    select_parser.add_argument(
        "--sequences-path",
        default=str(BPE_TOKEN_ID_SEQUENCES_FILE),
        help="BPE 토큰 시퀀스 파일 경로",
    )
    select_parser.add_argument(
        "--output-csv",
        default=str(REPLACEMENT_CANDIDATES_FILE),
        help="교체 후보 CSV 저장 경로",
    )
    select_parser.add_argument(
        "--output-log",
        default=str(SELECTION_LOG_FILE),
        help="선정 로그 저장 경로",
    )
    select_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="모델 이름",
    )
    select_parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="최대 후보 개수",
    )
    select_parser.add_argument(
        "--min-token-len",
        type=int,
        default=2,
        help="보호 토큰 최소 길이",
    )

    # remap 서브커맨드
    remap_parser = subparsers.add_parser("remap", help="토큰 재할당 규칙 적용")
    remap_parser.add_argument(
        "--distilled-tokenizer-dir",
        default=str(TOKENIZER_DISTILLED_UNIGRAM_DIR),
        help="증류된 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remapped-tokenizer-dir",
        default=str(TOKENIZER_REMAPPED_DIR),
        help="재할당 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remap-rules-path",
        default="src/gpt2_ivr/tokenizer/remap_rules.yaml",
        help="재할당 규칙 파일 경로",
    )
    remap_parser.add_argument(
        "--replacement-candidates-path",
        default=str(REPLACEMENT_CANDIDATES_FILE),
        help="교체 후보 CSV 경로",
    )

    # align 서브커맨드
    align_parser = subparsers.add_parser("align", help="embedding 재정렬")
    align_parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="GPT-2 모델 이름",
    )
    align_parser.add_argument(
        "--original-tokenizer-dir",
        default="artifacts/tokenizers/original",
        help="원본 토크나이저 디렉토리",
    )
    align_parser.add_argument(
        "--remapped-tokenizer-dir",
        default="artifacts/tokenizers/remapped",
        help="재할당 토크나이저 디렉토리",
    )
    align_parser.add_argument(
        "--remap-rules-path",
        default="src/gpt2_ivr/tokenizer/remap_rules.yaml",
        help="재할당 규칙 파일 경로",
    )
    align_parser.add_argument(
        "--embeddings-output-dir",
        default="artifacts/embeddings",
        help="임베딩 출력 디렉토리",
    )
    align_parser.add_argument(
        "--init-strategy",
        default="mean",
        choices=["mean", "random", "zeros"],
        help="신규 토큰 임베딩 초기화 전략",
    )

    # train 서브커맨드 (현재 stub)
    subparsers.add_parser("train", help="미세조정")

    return parser


def print_intro(logger: logging.Logger) -> None:
    """실행 개요를 출력한다."""
    logger.info("Tokenizer Model Migration + IVR 파이프라인")
    logger.info("BPE -> Unigram 토크나이저 교체 후 IVR를 수행합니다.")


def _create_init_command(args: argparse.Namespace) -> InitCommand:
    """init 커맨드를 생성한다."""
    return InitCommand(
        model_name=args.model_name,
        tokenizer_dir=Path(args.tokenizer_dir),
        force=args.force,
    )


def _create_analyze_command(args: argparse.Namespace) -> AnalyzeCommand:
    """analyze 커맨드를 생성한다."""
    return AnalyzeCommand(
        input_dir=Path(args.input_dir),
        output_sequences=Path(args.output_sequences),
        output_frequency=Path(args.output_frequency),
        model_name=args.model_name,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_texts=args.max_texts,
    )


def _create_distill_command(args: argparse.Namespace) -> DistillCommand:
    """distill-tokenizer 커맨드를 생성한다."""
    return DistillCommand(
        original_tokenizer_dir=Path(args.original_tokenizer_dir),
        distilled_tokenizer_dir=Path(args.distilled_tokenizer_dir),
        corpus_dir=Path(args.corpus_dir),
        vocab_size=args.vocab_size,
        model_name=args.model_name,
    )


def _create_select_command(args: argparse.Namespace) -> SelectCommand:
    """select 커맨드를 생성한다."""
    return SelectCommand(
        frequency_path=Path(args.frequency_path),
        sequences_path=Path(args.sequences_path),
        output_csv=Path(args.output_csv),
        output_log=Path(args.output_log),
        model_name=args.model_name,
        max_candidates=args.max_candidates,
        min_token_len=args.min_token_len,
    )


def _create_remap_command(args: argparse.Namespace) -> RemapCommand:
    """remap 커맨드를 생성한다."""
    return RemapCommand(
        distilled_tokenizer_dir=Path(args.distilled_tokenizer_dir),
        remapped_tokenizer_dir=Path(args.remapped_tokenizer_dir),
        remap_rules_path=Path(args.remap_rules_path),
        replacement_candidates_path=Path(args.replacement_candidates_path),
    )


def _create_align_command(args: argparse.Namespace) -> AlignCommand:
    """align 커맨드를 생성한다."""
    return AlignCommand(
        model_name=args.model_name,
        original_tokenizer_dir=Path(args.original_tokenizer_dir),
        remapped_tokenizer_dir=Path(args.remapped_tokenizer_dir),
        remap_rules_path=Path(args.remap_rules_path),
        embeddings_output_dir=Path(args.embeddings_output_dir),
        init_strategy=args.init_strategy,
    )


def _create_train_command(args: argparse.Namespace) -> TrainCommand:
    """train 커맨드를 생성한다. (현재 stub)"""
    # TODO: train 커맨드에 CLI 옵션이 추가되면 args를 사용하여 파라미터 전달
    return TrainCommand()


# 서브커맨드 팩토리 매핑
COMMAND_FACTORY_MAP = {
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
    if command_name in COMMAND_FACTORY_MAP:
        return COMMAND_FACTORY_MAP[command_name](args)

    raise NotImplementedError(f"'{command_name}'는 유효하지 않은 커맨드입니다.")


def dispatch(
    command_name: str, args: argparse.Namespace, logger: logging.Logger
) -> None:
    """서브커맨드를 실행한다."""
    logger.info("[%s] 단계를 시작합니다.", command_name)

    try:
        command = create_command(command_name, args)
        result = command.execute()
        logger.info("[%s] 단계 완료. 결과: %s", command_name, result)
    except NotImplementedError as e:
        logger.warning("%s", e)
    except Exception as e:
        logger.error("[%s] 실행 중 오류 발생: %s", command_name, e)
        raise


def main() -> None:
    """CLI 엔트리 포인트."""
    setup_logging()
    logger = get_logger("gpt2_ivr.cli")

    parser = build_parser()
    args = parser.parse_args()

    banner = build_banner("GPT2-IVR")
    print(banner)

    print_intro(logger)
    dispatch(args.command, args, logger)


if __name__ == "__main__":
    main()
