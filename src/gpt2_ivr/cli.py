from __future__ import annotations

import argparse
import logging

from pyfiglet import Figlet

from gpt2_ivr.commands import (
    AnalyzeCommand,
    Command,
    DistillCommand,
    SelectCommand,
    RemapCommand,
    AlignCommand,
    TrainCommand,
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

    # analyze 서브커맨드
    analyze_parser = subparsers.add_parser("analyze", help="BPE 토큰 시퀀스 분석")
    analyze_parser.add_argument(
        "--input-dir",
        default="artifacts/corpora/cleaned",
        help="코퍼스 입력 디렉토리",
    )
    analyze_parser.add_argument(
        "--output-sequences",
        default="artifacts/analysis/reports/bpe_token_id_sequences.txt",
        help="BPE 토큰 시퀀스 출력 경로",
    )
    analyze_parser.add_argument(
        "--output-frequency",
        default="artifacts/analysis/reports/token_frequency.parquet",
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
        default="artifacts/tokenizers/original",
        help="원본 토크나이저 디렉토리",
    )
    distill_parser.add_argument(
        "--distilled-tokenizer-dir",
        default="artifacts/tokenizers/distilled_unigram",
        help="증류된 토크나이저 저장 디렉토리",
    )
    distill_parser.add_argument(
        "--corpus-dir",
        default="artifacts/corpora/cleaned",
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
        default="artifacts/analysis/reports/token_frequency.parquet",
        help="토큰 빈도 parquet 파일 경로",
    )
    select_parser.add_argument(
        "--sequences-path",
        default="artifacts/analysis/reports/bpe_token_id_sequences.txt",
        help="BPE 토큰 시퀀스 파일 경로",
    )
    select_parser.add_argument(
        "--output-csv",
        default="artifacts/analysis/reports/replacement_candidates.csv",
        help="교체 후보 CSV 저장 경로",
    )
    select_parser.add_argument(
        "--output-log",
        default="artifacts/analysis/reports/selection_log.md",
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
        default="artifacts/tokenizers/distilled_unigram",
        help="증류된 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remapped-tokenizer-dir",
        default="artifacts/tokenizers/remapped",
        help="재할당 토크나이저 디렉토리",
    )
    remap_parser.add_argument(
        "--remap-rules-path",
        default="src/gpt2_ivr/tokenizer/remap_rules.yaml",
        help="재할당 규칙 파일 경로",
    )
    remap_parser.add_argument(
        "--replacement-candidates-path",
        default="artifacts/analysis/reports/replacement_candidates.csv",
        help="교체 후보 CSV 경로",
    )

    # align 서브커맨드 (현재 stub)
    subparsers.add_parser("align", help="embedding 재정렬")

    # train 서브커맨드 (현재 stub)
    subparsers.add_parser("train", help="미세조정")

    return parser


def print_intro(logger: logging.Logger) -> None:
    """실행 개요를 출력한다."""
    logger.info("Tokenizer Model Migration + IVR 파이프라인")
    logger.info("BPE -> Unigram 토크나이저 교체 후 IVR를 수행합니다.")


def create_command(command_name: str, args: argparse.Namespace) -> Command:
    """커맨드 이름에 해당하는 Command 객체를 생성한다."""
    from pathlib import Path

    if command_name == "analyze":
        return AnalyzeCommand(
            input_dir=Path(args.input_dir),
            output_sequences=Path(args.output_sequences),
            output_frequency=Path(args.output_frequency),
            model_name=args.model_name,
            workers=args.workers,
            chunk_size=args.chunk_size,
            max_texts=args.max_texts,
        )
    elif command_name == "distill-tokenizer":
        return DistillCommand(
            original_tokenizer_dir=Path(args.original_tokenizer_dir),
            distilled_tokenizer_dir=Path(args.distilled_tokenizer_dir),
            corpus_dir=Path(args.corpus_dir),
            vocab_size=args.vocab_size,
            model_name=args.model_name,
        )
    elif command_name == "select":
        return SelectCommand(
            frequency_path=Path(args.frequency_path),
            sequences_path=Path(args.sequences_path),
            output_csv=Path(args.output_csv),
            output_log=Path(args.output_log),
            model_name=args.model_name,
            max_candidates=args.max_candidates,
            min_token_len=args.min_token_len,
        )
    elif command_name == "remap":
        return RemapCommand(
            distilled_tokenizer_dir=Path(args.distilled_tokenizer_dir),
            remapped_tokenizer_dir=Path(args.remapped_tokenizer_dir),
            remap_rules_path=Path(args.remap_rules_path),
            replacement_candidates_path=Path(args.replacement_candidates_path),
        )
    elif command_name == "align":
        return AlignCommand()
    elif command_name == "train":
        return TrainCommand()
    else:
        raise NotImplementedError(f"'{command_name}'는 유효하지 않은 커맨드입니다.")


def dispatch(command_name: str, args: argparse.Namespace, logger: logging.Logger) -> None:
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
